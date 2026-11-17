"""Audited cross-arm findings, using only actual recorded model outputs and usage."""
from pathlib import Path
from collections import Counter
import csv
import math
import statistics
from .common import CAMPAIGN, METHODS, atomic, read
from ourwork.baseline.records import read_jsonl, fingerprint
from ourwork.baseline.metrics import answer_scores, hypervolume, igd, pareto_indices
from ourwork.baseline.analyze import write_csv

ARM_FIELDS = {'optimizer_method', 'optimizer_implementation_sha256',
              'shared_initial_pool_sha256', 'population_size'}


def comparable(protocol):
    return {k: v for k, v in protocol['experiment'].items() if k not in ARM_FIELDS}


def validate_protocols(protocols, questions, grounding, packages):
    reference = protocols[0]
    for p, q, g, env in zip(protocols[1:], questions[1:], grounding[1:], packages[1:]):
        if comparable(p) != comparable(reference):
            raise ValueError('Different common evaluation protocols')
        for key in ['optimizer_seed', 'source_identity', 'sources', 'server']:
            if p[key] != reference[key]:
                raise ValueError(f'Cross-arm mismatch: {key}')
        if p['experiment']['optimizer_implementation_sha256'] != reference['experiment']['optimizer_implementation_sha256']:
            raise ValueError('Different campaign/optimizer implementations')
        if q != questions[0] or g != grounding[0] or env != packages[0]:
            raise ValueError('Different questions, grounding corpus, or package environment')


def audit_trial(run, trial, manifest, protocol, sources=None):
    """Recompute primary objectives from raw answers and endpoint call journals."""
    dest = run / trial.user_attrs['result_path']
    job = read(dest / 'job.json')
    params = trial.user_attrs['concrete_params']
    if job['params'] != params or fingerprint(params) != trial.user_attrs['candidate_id']:
        raise ValueError('Recorded candidate parameters/identity disagree')
    if job['question_manifest'] != manifest:
        raise ValueError('Worker question manifest differs')
    if comparable({'experiment': job['protocol']}) != protocol:
        raise ValueError('Worker used a different evaluation protocol')
    if sources is not None and job['sources'] != sources:
        raise ValueError('Worker source/model/data pins differ')
    rows = read_jsonl(dest / 'questions.jsonl')
    if len(rows) != protocol['qa_count'] or len({r['qa_id'] for r in rows}) != len(rows):
        raise ValueError('Eligible candidate is missing unique QA rows')
    if [r['qa_id'] for r in rows] != manifest['selected_ids']:
        raise ValueError('Candidate evaluated different questions/order')
    if sum(r['generation_error'] is None for r in rows) / len(rows) < protocol['minimum_success_rate']:
        raise ValueError('Candidate violates the declared generation success gate')
    values = []
    for row in rows:
        score = answer_scores(row['answer'], row['reference'])
        if row['generation_error']:
            score = {'f1': 0.0, 'exact_match': 0.0}
        if any(not math.isclose(score[k], row[k], abs_tol=1e-10) for k in score):
            raise ValueError('Saved F1/EM disagrees with raw answer/reference')
        values.append(score['f1'])
    calls = read_jsonl(dest / 'calls.jsonl')
    generation = [c for c in calls if c['role'] == 'generation']
    requested = [c for c in read_jsonl(dest / 'requests.jsonl') if c['role'] == 'generation']
    if not generation or {c['call_id'] for c in generation} != {c['call_id'] for c in requested}:
        raise ValueError('Generation request/completion coverage is incomplete')
    if any(not c['usage_verified'] or c['http_status'] != 200 for c in generation):
        raise ValueError('Eligible candidate has failed or unknown-usage generation calls')
    quality = statistics.mean(values)
    resource = sum(c['usage']['prompt_tokens'] + c['usage']['completion_tokens'] for c in generation) / len(rows)
    if not math.isclose(quality, trial.values[0], abs_tol=1e-10) or not math.isclose(resource, trial.values[1], abs_tol=1e-8):
        raise ValueError('Trial objectives disagree with recomputed raw measurements')
    return rows


def coverage(left, right):
    return (sum(any(q >= q2 and c <= c2 for q, c in left) for q2, c2 in right) / len(right)
            if right else None)


def compare(campaign=CAMPAIGN, make_plots=True, prefix=None):
    import optuna
    campaign = Path(campaign)
    runs = [campaign / 'runs' / f'{m}-seed11' for m in METHODS]
    protocols = [read(p / 'protocol.json') for p in runs]
    questions = [read(p / 'questions_manifest.json') for p in runs]
    grounding = [read(p / 'grounding_manifest.json') for p in runs]
    packages = [read(p / 'packages.json') for p in runs]
    validate_protocols(protocols, questions, grounding, packages)
    profile = comparable(protocols[0])
    scale = profile['resource_reference']
    prefix = profile['budget'] if prefix is None else prefix
    if not profile['startup_trials'] <= prefix <= profile['budget']:
        raise ValueError('Comparison checkpoint is outside the declared campaign')
    trials_by_arm, points_by_arm, front_by_arm, summaries = {}, {}, {}, []
    all_qa, initial_prefixes = [], []
    for method, path, wrapped, manifest in zip(METHODS, runs, protocols, questions):
        if wrapped['experiment']['optimizer_method'] != method:
            raise ValueError('Run label differs from declared optimizer')
        summary = read(path / 'analysis/summary.json')
        study = optuna.load_study(study_name='baseline', storage=f'sqlite:///{path / "study.sqlite3"}')
        trials = [t for t in study.trials if t.state != optuna.trial.TrialState.WAITING]
        if len(trials) < prefix or len(trials) > profile['budget'] or any(not t.state.is_finished() for t in trials[:prefix]):
            raise ValueError('Every arm must finish the same declared comparison prefix')
        trials = trials[:prefix]
        with (path / 'analysis/convergence.csv').open() as f:
            checkpoint_curve = list(csv.DictReader(f))[:prefix]
        checkpoint = checkpoint_curve[-1]
        summary = dict(summary, attempts=prefix, completed=sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials),
            failed=sum(t.state == optuna.trial.TrialState.FAIL for t in trials), budget_exhausted=prefix == profile['budget'],
            **{k: int(checkpoint[k]) for k in ['known_generator_tokens','known_judge_tokens','model_calls','unknown_usage_calls']})
        initial_prefixes.append([t.user_attrs.get('concrete_params') for t in trials[:profile['startup_trials']]])
        eligible = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        points = [tuple(t.values) for t in eligible]
        for t in eligible:
            rows = audit_trial(path, t, manifest, profile, wrapped['sources'])
            all_qa.extend(dict(r, method=method, trial=t.number) for r in rows)
        trials_by_arm[method] = eligible
        points_by_arm[method] = points
        front_by_arm[method] = [points[i] for i in pareto_indices(points)]
        metrics = [t.user_attrs['metrics'] for t in eligible]
        row = dict(summary, method=method, seed=wrapped['optimizer_seed'],
                   best_f1=max((q for q, c in points), default=None),
                   min_tokens_per_qa=min((c for q, c in points), default=None),
                   mean_eligible_f1=statistics.mean(q for q, c in points) if points else None,
                   mean_eligible_em=statistics.mean(m['exact_match'] for m in metrics) if metrics else None,
                   total_candidate_wall_seconds=sum(t.user_attrs.get('wall_seconds', 0) for t in trials),
                   total_construction_seconds=sum(m.get('construction_seconds', 0) for m in metrics),
                   eligible_truncated_questions=sum(m.get('truncated_questions', 0) for m in metrics),
                   raw_objective_audit='passed',
                   failure_reasons=dict(Counter(t.user_attrs.get('failure_reason', 'unspecified')
                       for t in trials if t.state == optuna.trial.TrialState.FAIL)))
        row['hypervolume'] = hypervolume(points, scale)
        row['pareto_count'] = len(pareto_indices(points))
        row['unique_pareto_objective_pairs'] = len(set(front_by_arm[method]))
        for token_limit in [500, 1000, 2000, 5000, 10000, 50000]:
            row[f'best_f1_at_tokens_le_{token_limit}'] = max((q for q, c in points if c <= token_limit), default=None)
        for quality_floor in [.25, .5, .75]:
            row[f'min_tokens_at_f1_ge_{quality_floor}'] = min((c for q, c in points if q >= quality_floor), default=None)
        with (path / 'analysis/convergence.csv').open() as f:
            curve = list(csv.DictReader(f))
        row['mean_end_of_attempt_hv'] = sum(float(r['hypervolume']) for r in curve[:prefix]) / prefix
        row['workflow_attempts'] = dict(Counter(t.user_attrs.get('concrete_params', {}).get('rag_mode', 'unknown') for t in trials))
        row['workflow_eligible'] = dict(Counter(t.user_attrs['concrete_params']['rag_mode'] for t in eligible))
        summaries.append(row)
    if any(pool != initial_prefixes[0] for pool in initial_prefixes[1:]):
        raise ValueError('Starting configurations are not matched across methods')
    pooled = [p for points in points_by_arm.values() for p in points]
    reference = [pooled[i] for i in pareto_indices(pooled)]
    out = campaign / ('comparison' if prefix == profile['budget'] else f'comparison_at_{prefix}')
    out.mkdir(exist_ok=True)
    front_rows, selections = [], []
    for row in summaries:
        method = row['method']
        points = points_by_arm[method]
        row['igd_to_pooled_empirical_front'] = igd(points, reference, scale)
        others = [p for m, ps in points_by_arm.items() if m != method for p in ps]
        row['exclusive_hv_contribution'] = hypervolume(pooled, scale) - hypervolume(others, scale)
        ts = trials_by_arm[method]
        front_rows.extend({'method': method, 'trial': ts[i].number,
                           'f1': points[i][0], 'generator_tokens_per_qa': points[i][1],
                           'params': ts[i].user_attrs['concrete_params']}
                          for i in pareto_indices(points))
        # Frozen before confirmation: maximize single-point normalized HV, then F1,
        # then lower tokens, then earliest trial. No holdout outcome is consulted.
        if ts:
            selected = max(ts, key=lambda t: (hypervolume([tuple(t.values)], scale),
                                               t.values[0], -t.values[1], -t.number))
            selections.append({'method': method, 'trial': selected.number,
                'source_run': str(campaign / 'runs' / f'{method}-seed11'),
                'rule': 'max_single_point_hv_then_f1_then_lower_tokens_then_earliest_trial'})
    pairwise = [{'method_a': a, 'method_b': b, 'coverage_a_over_b': coverage(front_by_arm[a], front_by_arm[b])}
                for a in METHODS for b in METHODS if a != b]
    write_csv(out / 'methods.csv', summaries)
    write_csv(out / 'pareto_fronts.csv', front_rows)
    write_csv(out / 'pairwise_front_coverage.csv', pairwise)
    write_csv(out / 'eligible_per_question.csv', all_qa)
    atomic(out / 'findings.json', {'status': 'audited_search_results' if prefix == profile['budget'] else 'planned_intermediate_comparison',
        'comparison_attempts': prefix, 'declared_final_budget': profile['budget'], 'methods': summaries,
        'empirical_reference': reference, 'fronts': front_rows, 'points': points_by_arm, 'paired_coverage': pairwise,
        'limits': ['One optimizer seed and ten search questions: exploratory evidence only.',
                   'Pooled empirical IGD is not distance to the unknown global optimum.',
                   'Shared retrieval caches and fixed arm order confound wall-clock comparisons.',
                   'Local judge accuracy is diagnostic only; token cost excludes embedding FLOPs and API dollars.']})
    selection_file = out / 'confirmation_selection.json'
    if selection_file.exists() and read(selection_file) != selections:
        raise ValueError('Previously frozen confirmation selection changed')
    if prefix == profile['budget']:
        atomic(selection_file, selections)
    lines = ['# Audited four-method search comparison', '',
             f'Checkpoint: {prefix} attempted candidates per arm; declared final budget {profile["budget"]}.',
             'At checkpoint 10, only shared initialization has run: no algorithm-specific search advantage can be inferred.' if prefix == profile['startup_trials'] else '', '',
             'One pinned Qwen model; restricted Syftr subset; seed 11. Search metrics are not independent test evidence.', '',
             '| Method | Attempts | Eligible | Failed | HV | Best F1 | Cheapest tokens/QA |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f'| {r["method"]} | {r["attempts"]} | {r["completed"]} | {r["failed"]} | {r["hypervolume"]:.6f} | {r["best_f1"]} | {r["min_tokens_per_qa"]} |')
    lines += ['', 'All eligible objectives were recomputed from raw answers and generation request/completion journals.', '',
              'Use methods.csv, pareto_fronts.csv, pairwise_front_coverage.csv, per-run convergence.csv, '
              'module outcomes/associations, failure logs, generation logs and raw per-question records for findings.', '',
              'Coverage counts weak dominance, including identical points. More sampled good configurations is not a causal module-effect claim.', '',
              'Timing is descriptive under a warm shared index cache; no unbiased wall-clock speedup claim. '
              'No across-seed significance claim is supported. Holdout confirmation is separate.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if make_plots:
        plot(runs, points_by_arm, front_by_arm, out, prefix)
    return summaries


def plot(runs, points, fronts, out, prefix=None):
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'syftr-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ['pareto', 'convergence_attempts', 'convergence_tokens']:
        fig, ax = plt.subplots(figsize=(8, 5))
        for color_index, (method, run) in enumerate(zip(METHODS, runs)):
            color = plt.get_cmap("tab10")(color_index)
            if kind == 'pareto':
                all_points = points[method]
                ax.scatter([c for q, c in all_points], [q for q, c in all_points], alpha=.25, s=18, color=color)
                front = sorted(set(fronts[method]), key=lambda x: x[1])
                ax.plot([c for q, c in front], [q for q, c in front], marker='o', label=method, color=color)
            else:
                with (run / 'analysis/convergence.csv').open() as f:
                    curve = list(csv.DictReader(f))[:prefix][:prefix]
                x = 'attempt' if kind == 'convergence_attempts' else 'known_generator_tokens'
                ax.step([float(r[x]) for r in curve], [float(r['hypervolume']) for r in curve], where='post', label=method, color=color)
        ax.set_xlabel({'pareto': 'Mean generator prompt + completion tokens / QA',
                       'convergence_attempts': 'Attempted candidates (failures included)',
                       'convergence_tokens': 'Known cumulative generator search tokens'}[kind])
        ax.set_ylabel('Raw-answer F1' if kind == 'pareto' else 'Normalized hypervolume')
        ax.set_title('Restricted Syftr subset: search set, seed 11')
        ax.set_ylim(-.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        for extension in ['png', 'pdf']:
            fig.savefig(out / f'{kind}.{extension}', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    import sys
    prefix = int(sys.argv[sys.argv.index('--prefix') + 1]) if '--prefix' in sys.argv else None
    if '--plots-only' in sys.argv:
        budget = read(CAMPAIGN / 'protocol.json')['budget']
        out = CAMPAIGN / ('comparison' if prefix in {None, budget} else f'comparison_at_{prefix}')
        findings = read(out / 'findings.json')
        points = findings['points']
        fronts = {m: [(r['f1'], r['generator_tokens_per_qa']) for r in findings['fronts'] if r['method'] == m]
                  for m in METHODS}
        plot([CAMPAIGN / 'runs' / f'{m}-seed11' for m in METHODS], points, fronts, out, prefix)
    else:
        compare(prefix=prefix)
