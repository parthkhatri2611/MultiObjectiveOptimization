"""One preselected candidate per arm on disjoint holdout QA; no holdout search."""
import argparse
import fcntl
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace
from .common import CAMPAIGN, OUTPUT, ROOT, atomic, read, workspace_for, checked_source_identity
from .snapshots import snapshot
from .report import audit_trial, comparable


def paired_bootstrap(left, right, iterations=2000):
    if set(left) != set(right) or not left:
        raise ValueError('Paired confirmation requires identical nonempty QA identifiers')
    ids = sorted(left)
    differences = [left[q] - right[q] for q in ids]
    rng = random.Random(20261116)
    values = sorted(statistics.mean(rng.choices(differences, k=len(ids))) for _ in range(iterations))
    return {'mean_f1_difference': statistics.mean(differences),
            'question_bootstrap_95_percentile_interval': [values[int(.025 * iterations)], values[int(.975 * iterations)]],
            'qa_count': len(ids), 'iterations': iterations,
            'scope': 'Question resampling conditional on these selected configurations; not optimizer-seed uncertainty.'}


def confirm(hours=1):
    from ourwork.baseline.adapter import prepare, question_manifest, grounding_manifest
    from ourwork.baseline.records import source_identity, fingerprint, read_jsonl
    from ourwork.baseline.run import execute_job, eligible
    from ourwork.baseline.analyze import write_csv
    import optuna
    lock = (CAMPAIGN / '.controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selections = read(CAMPAIGN / 'comparison/confirmation_selection.json')
    out = CAMPAIGN / 'confirmation'
    out.mkdir(exist_ok=True)
    deadline = time.monotonic() + hours * 3600
    summaries, answer_sets = [], {}
    try:
        for selection in selections:
            atomic(out / 'status.json', {'status': 'running', 'active_method': selection['method'],
                                        'completed_arms': len(summaries), 'qa_count': 10})
            source = Path(selection['source_run'])
            wrapped = read(source / 'protocol.json')
            actual_identity = source_identity(ROOT)
            checked_source_identity(actual_identity, wrapped['source_identity'])
            atomic(out / ('actual_common_source_identity_' + selection['method'] + '.json'), actual_identity)
            if read(OUTPUT / 'server_metadata.json') != wrapped['server']:
                raise ValueError('Common source/server changed before confirmation')
            protocol = dict(wrapped['experiment'], partition='holdout', stage='confirm', qa_count=10)
            workspace = workspace_for(selection['method'], 11)
            study_config, _, _ = prepare(protocol, workspace, wrapped['sources'])
            _, manifest = question_manifest(study_config, protocol)
            search_manifest = read(source / 'questions_manifest.json')
            if set(manifest['selected_ids']) & set(search_manifest['selected_ids']):
                raise ValueError('Holdout questions overlap searched questions')
            study = optuna.load_study(study_name='baseline', storage=f'sqlite:///{source / 'study.sqlite3'}')
            trial = next(t for t in study.trials if t.number == selection['trial'])
            if trial.state != optuna.trial.TrialState.COMPLETE:
                raise ValueError('Selected candidate is not eligible')
            params = trial.user_attrs['concrete_params']
            cid = fingerprint(params)
            destination = out / selection['method']
            immutable = {'selection': selection, 'experiment': protocol,
                         'questions_manifest': manifest, 'source_protocol': wrapped,
                         'grounding': grounding_manifest(study_config)}
            if (destination / 'protocol.json').exists() and read(destination / 'protocol.json') != immutable:
                raise ValueError('Saved holdout selection/protocol changed')
            atomic(destination / 'protocol.json', immutable)
            result_path = destination / 'result.json'
            if result_path.exists():
                result = read(result_path)
            else:
                if (CAMPAIGN / 'STOP_REQUESTED').exists() or deadline - time.monotonic() < 120:
                    atomic(out / 'status.json', {'status': 'paused', 'completed_arms': len(summaries)})
                    return
                job = {'protocol': protocol, 'params': params, 'sources': wrapped['sources'],
                       'question_manifest': manifest, 'candidate_id': cid,
                       'evaluation_seed': int(cid[:8], 16), 'workspace': str(workspace),
                       'destination': str(destination)}
                result = execute_job(job, min(protocol['candidate_timeout'], deadline - time.monotonic() - 60), ROOT)
                if not result_path.exists():
                    atomic(result_path, result)
            valid = result.get('status') == 'evaluated' and eligible(result.get('metrics', {}), protocol)
            summaries.append({'method': selection['method'], 'source_trial': trial.number,
                              'eligible': valid, **result.get('metrics', {}), 'error': result.get('error')})
            if valid:
                measured = result['metrics']
                audit_record = SimpleNamespace(user_attrs={'result_path': selection['method'],
                    'concrete_params': params, 'candidate_id': cid},
                    values=[measured['f1'], measured['total_tokens_mean']])
                rows = audit_trial(out, audit_record, manifest,
                                   comparable({'experiment': protocol}), wrapped['sources'])
                answer_sets[selection['method']] = {r['qa_id']: r['f1'] for r in rows}
            atomic(out / 'results.json', summaries)
            snapshot(CAMPAIGN)
        differences = {m: paired_bootstrap(values, answer_sets['syftr_motpe'])
                       for m, values in answer_sets.items() if m != 'syftr_motpe'} if 'syftr_motpe' in answer_sets else {}
        atomic(out / 'paired_f1_differences.json', differences)
        write_csv(out / 'methods.csv', summaries)
        atomic(out / 'status.json', {'status': 'complete', 'arms_attempted': len(summaries),
                                     'eligible_arms': len(answer_sets), 'independent_qa_count': 10})
    finally:
        snapshot(CAMPAIGN)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=1)
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error('Hours must be positive')
    confirm(args.hours)
