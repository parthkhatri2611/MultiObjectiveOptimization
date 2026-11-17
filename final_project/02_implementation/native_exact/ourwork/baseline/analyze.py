"""Rebuild findings from real run artifacts. No invented benchmark values."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
from .metrics import hypervolume, pareto_indices, igd
from .records import atomic_json, read_jsonl


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row})
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def analyze(path):
    import optuna
    path = Path(path)
    protocol = json.loads((path / 'protocol.json').read_text())['experiment']
    study = optuna.load_study(study_name='baseline', storage=f'sqlite:///{path / "study.sqlite3"}')
    rows, points, complete = [], [], []
    grouped = defaultdict(list)
    for t in study.trials:
        if t.state == optuna.trial.TrialState.WAITING:
            continue
        metrics = t.user_attrs.get('metrics', {})
        row = {'trial': t.number, 'state': t.state.name, 'candidate_id': t.user_attrs.get('candidate_id'),
               'quality': t.values[0] if t.values else None, 'resource': t.values[1] if t.values else None,
               'wall_seconds': t.user_attrs.get('wall_seconds'),
               'failure_reason': t.user_attrs.get('failure_reason'),
               'capability': t.user_attrs.get('capability'),
               'params': t.user_attrs.get('concrete_params', t.params), **metrics}
        rows.append(row)
        if t.state == optuna.trial.TrialState.COMPLETE and t.values:
            complete.append(row)
            points.append(tuple(t.values))
            for key, value in row['params'].items():
                grouped[(key, str(value))].append(row)
    front = [complete[i] for i in pareto_indices(points)]
    front_ids = {r['candidate_id'] for r in front}
    output = path / 'analysis'
    output.mkdir(exist_ok=True)
    write_csv(output / 'trials.csv', rows)
    write_csv(output / 'pareto.csv', front)
    atomic_json(output / 'pareto.json', front)
    observed, curves = [], []
    generator_tokens, judge_tokens, calls, unknown_usage_calls, elapsed = 0, 0, 0, 0, 0.0
    # Include work from interrupted/timed-out candidates, not just completed metrics.
    for row in rows:
        candidate_dirs = list((path / 'candidates').glob(f'{row["trial"]:05d}-*')) if (path / 'candidates').exists() else []
        for candidate in candidate_dirs:
            completed_calls = read_jsonl(candidate / 'calls.jsonl')
            requests = read_jsonl(candidate / 'requests.jsonl')
            completed_ids = {c.get('call_id') for c in completed_calls}
            calls += len(requests) if requests else len(completed_calls)
            unknown_usage_calls += sum(r.get('call_id') not in completed_ids for r in requests)
            unknown_usage_calls += sum(not c.get('usage_verified') for c in completed_calls)
            for call in completed_calls:
                count = call.get('usage', {}).get('prompt_tokens', 0) + call.get('usage', {}).get('completion_tokens', 0)
                if call.get('role') == 'judge':
                    judge_tokens += count
                else:
                    generator_tokens += count
        elapsed += row.get('wall_seconds') or 0
        if row['state'] == 'COMPLETE':
            observed.append((row['quality'], row['resource']))
        curves.append({'attempt': len(curves) + 1, 'trial': row['trial'],
                       'completed_candidates': len(observed), 'known_generator_tokens': generator_tokens,
                       'known_judge_tokens': judge_tokens, 'model_calls': calls,
                       'unknown_usage_calls': unknown_usage_calls,
                       'candidate_wall_seconds': elapsed,
                       'hypervolume': hypervolume(observed, protocol['resource_reference'])})
    write_csv(output / 'convergence.csv', curves)
    params = []
    for (key, value), members in sorted(grouped.items()):
        params.append({'parameter': key, 'value': value, 'completed_count': len(members),
                       'mean_quality': statistics.mean(r['quality'] for r in members),
                       'mean_resource': statistics.mean(r['resource'] for r in members),
                       'pareto_count': sum(r['candidate_id'] in front_ids for r in members)})
    write_csv(output / 'parameter_observations.csv', params)
    outcomes = defaultdict(list)
    for row in rows:
        for key, value in row['params'].items():
            outcomes[(key, str(value))].append(row)
    write_csv(output / 'component_outcomes.csv', [
        {'parameter': key, 'value': value, 'attempts': len(members),
         'complete': sum(r['state'] == 'COMPLETE' for r in members),
         'failed': sum(r['state'] == 'FAIL' for r in members)}
        for (key, value), members in sorted(outcomes.items())])
    failures = Counter(row.get('failure_reason') or 'unspecified' for row in rows if row['state'] == 'FAIL')
    atomic_json(output / 'failures.json', dict(failures))
    question_rows = []
    for p in sorted((path / 'candidates').glob('*/questions.jsonl')) if (path / 'candidates').exists() else []:
        question_rows.extend(read_jsonl(p))
    write_csv(output / 'per_question.csv', question_rows)
    # Flags identify candidates for manual inspection, not confirmed judge mistakes.
    write_csv(output / 'human_audit.csv', [
        {k: r.get(k) for k in ['candidate_id', 'qa_id', 'question', 'reference', 'answer',
                               'score', 'passing', 'f1', 'exact_match', 'generation_error', 'judge_error']}
        for r in question_rows if (r['passing'] and r['f1'] < .2) or (not r['passing'] and r['f1'] > .9)])
    question_groups = defaultdict(list)
    for r in question_rows:
        for field in ['qtype', 'difficulty']:
            question_groups[(field, r.get(field, 'unknown'))].append(r)
    write_csv(output / 'question_groups.csv', [
        {'group_field': field, 'group': value, 'observations': len(members),
         'accuracy': sum(bool(r['passing']) for r in members) / len(members),
         'mean_f1': statistics.mean(r['f1'] for r in members),
         'mean_generation_seconds': statistics.mean(r['generation_seconds'] for r in members)}
        for (field, value), members in question_groups.items()])
    summary = {'status': 'measured' if complete else 'no_eligible_measured_candidates',
               'attempts': len(rows), 'completed': len(complete), 'failed': sum(r['state'] == 'FAIL' for r in rows),
               'budget': protocol['budget'], 'budget_exhausted': len(rows) >= protocol['budget'],
               'pareto_count': len(front), 'hypervolume': hypervolume(points, protocol['resource_reference']),
               'unique_pareto_objective_pairs': len({(r['quality'], r['resource']) for r in front}),
               'objective_quality': protocol['objective_quality'], 'objective_resource': protocol['objective_resource'],
               'resource_reference': protocol['resource_reference'], 'known_generator_tokens': generator_tokens,
               'known_judge_tokens': judge_tokens, 'model_calls': calls,
               'unknown_usage_calls': unknown_usage_calls,
               'capability_results': {r['capability']: r['state'] for r in rows if r.get('capability')}}
    atomic_json(output / 'summary.json', summary)
    lines = ['# Measured Syftr local-model baseline findings', '',
             f"Stage: **{protocol['stage']}**. Dataset: **{protocol['dataset']}**, partition **{protocol['partition']}**.",
             f"Quality: `{protocol['objective_quality']}`; resource: `{protocol['objective_resource']}`.", '',
             f"Attempted {len(rows)} / {protocol['budget']} candidates; {len(complete)} eligible complete evaluations; {len(front)} Pareto configurations.",
             f"Normalized hypervolume: {summary['hypervolume']:.6f}, fixed resource reference {protocol['resource_reference']}.", '',
             'This report contains only saved measurements. Failed/partial candidates are excluded from the final frontier but their known calls/tokens remain in search-cost accounting.', '',
             '## Supported observations', '',
             '- `pareto.csv`: complete trade-offs and exact configurations.',
             '- `convergence.csv`: hypervolume versus attempted candidates, known calls/tokens and candidate wall time.',
             '- `parameter_observations.csv`: conditional observed component frequencies and average objectives; these are associations, not causal improvements.',
             '- `failures.json`: infrastructure/model/measurement failures, retained in the budget.',
             '- Each candidate contains raw answers, judge outcomes, truncation, per-query timing and nested model request/response usage.', '',
             '## Interpretation limits', '',
             '- The generator and local judge share weights: judge bias remains; inspect calibration and human-audit examples.',
             '- Scores and resource use are local adaptations, not reproductions of the paper\'s hosted-model dollar figures.',
             '- Search-set outcomes require independent holdout confirmation.',
             '- A baseline alone cannot establish whether evolutionary approaches are better.',
             '- Comparison runs must share this exact data/model/search-space/objective protocol and hardware block.', '']
    if not complete:
        lines += ['**No eligible complete candidates exist. No performance conclusion is available.**', '']
    else:
        best = max(complete, key=lambda r: r['quality'])
        cheapest = min(complete, key=lambda r: r['resource'])
        lines += [f"Highest observed quality: {best['quality']:.6f} at resource {best['resource']:.3f} (trial {best['trial']}).",
                  f"Lowest observed resource: {cheapest['resource']:.3f} at quality {cheapest['quality']:.6f} (trial {cheapest['trial']}).", '']
    (output / 'FINDINGS.md').write_text('\n'.join(lines))
    if points:
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter([c for q, c in points], [q for q, c in points], alpha=.5, label='Complete candidates')
            pf = sorted([(r['resource'], r['quality']) for r in front])
            ax.plot([c for c, q in pf], [q for c, q in pf], 'o-', label='Observed Pareto frontier')
            ax.set(xlabel=protocol['objective_resource'], ylabel=protocol['objective_quality'], title='Measured local Syftr baseline')
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / 'pareto.png', dpi=160)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([r['attempt'] for r in curves], [r['hypervolume'] for r in curves])
            ax.set(xlabel='Candidate attempts (initialization and failures included)', ylabel='Normalized hypervolume')
            fig.tight_layout()
            fig.savefig(output / 'convergence.png', dpi=160)
            plt.close(fig)
        except ImportError:
            pass  # CSV/JSON findings remain usable in the lightweight local environment.
    return summary


def aggregate(paths, output):
    # Pooled empirical reference is explicitly not the unknown global frontier.
    fronts, all_points = {}, []
    for path in paths:
        analyze(path)
        front = json.loads((Path(path) / 'analysis/pareto.json').read_text())
        points = [(r['quality'], r['resource']) for r in front]
        fronts[str(path)] = points
        all_points.extend(points)
    protocols = [json.loads((Path(p) / 'protocol.json').read_text()) for p in paths]
    experiments = [p['experiment'] for p in protocols]
    if any(e != experiments[0] for e in experiments[1:]):
        raise ValueError('Cannot aggregate different protocols')
    if any(p['source_identity'] != protocols[0]['source_identity'] or p['sources'] != protocols[0]['sources'] for p in protocols[1:]):
        raise ValueError('Cannot aggregate differing code/model/data revisions')
    if any(p.get('server') != protocols[0].get('server') for p in protocols[1:]):
        raise ValueError('Cannot aggregate different serving implementations/settings')
    reference = [all_points[i] for i in pareto_indices(all_points)]
    scale = experiments[0]['resource_reference']
    records = [{'run': name, 'hv': hypervolume(points, scale),
                'igd_to_pooled_empirical_reference': igd(points, reference, scale)} for name, points in fronts.items()]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'runs.csv', records)
    atomic_json(out / 'empirical_reference.json', reference)
    atomic_json(out / 'aggregate.json', {'runs': len(records),
                'mean_hv': statistics.mean(r['hv'] for r in records),
                'std_hv': statistics.stdev(r['hv'] for r in records) if len(records) > 1 else None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--aggregate-output')
    args = parser.parse_args()
    if args.aggregate_output:
        aggregate(args.runs, args.aggregate_output)
    else:
        for path in args.runs:
            print(json.dumps(analyze(path), indent=2))
