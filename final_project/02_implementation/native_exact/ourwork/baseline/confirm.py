"""Evaluate preselected search candidates on holdout, without proposing or selecting there."""
import argparse
import json
from pathlib import Path
import time
from .adapter import prepare, question_manifest, grounding_manifest
from .records import RunStore, atomic_json, environment, fingerprint, source_identity
from .run import execute_job
from .spec import load_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--trials', required=True, help='Comma-separated trial numbers selected using search data only')
    parser.add_argument('--qa-count', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--server-metadata', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import optuna
    source = Path(args.source_run).resolve()
    original = json.loads((source / 'protocol.json').read_text())
    root = Path(__file__).resolve().parents[2]
    if source_identity(root) != original['source_identity']:
        raise ValueError('Source code changed since the search run')
    metadata = json.loads(Path(args.server_metadata).read_text())
    if metadata != original['server']:
        raise ValueError('Model/server differs from search')
    study = optuna.load_study(study_name='baseline', storage=f'sqlite:///{source / "study.sqlite3"}')
    chosen = [int(n) for n in args.trials.split(',')]
    if len(chosen) != len(set(chosen)) or args.qa_count < 1:
        raise ValueError('Trial numbers must be unique and QA count positive')
    selection = []
    for n in chosen:
        trial = next(t for t in study.trials if t.number == n)
        if trial.state != optuna.trial.TrialState.COMPLETE:
            raise ValueError('Only fully evaluated search candidates can be confirmed')
        selection.append({'source_trial': n, 'params': trial.user_attrs['concrete_params'], 'search_values': trial.values})
    protocol = {**original['experiment'], 'partition': 'holdout', 'stage': 'confirm', 'qa_count': args.qa_count}
    store = RunStore(args.output, {'source_protocol': original, 'experiment': protocol, 'preselected_candidates': selection}, args.resume)
    atomic_json(store.path / 'environment.json', environment())
    fixed_study, _, _ = prepare(protocol, args.workspace, original['sources'])
    _, manifest = question_manifest(fixed_study, protocol)
    atomic_json(store.path / 'questions_manifest.json', manifest)
    atomic_json(store.path / 'grounding_manifest.json', grounding_manifest(fixed_study))
    results = []
    for item in selection:
        cid = fingerprint(item['params'])
        dest = store.path / 'candidates' / f'{item["source_trial"]:05d}-{cid[:12]}'
        if args.resume and (dest / 'result.json').exists():
            result = json.loads((dest / 'result.json').read_text())
        else:
            result = execute_job({'protocol': protocol, 'params': item['params'], 'sources': original['sources'],
                                  'question_manifest': manifest, 'candidate_id': cid,
                                  'evaluation_seed': int(cid[:8], 16), 'workspace': str(Path(args.workspace).resolve()),
                                  'destination': str(dest.resolve())}, protocol['candidate_timeout'], root)
        results.append({**item, 'holdout_result': result})
        atomic_json(store.path / 'holdout_results.json', results)
    store.event('confirmation_finished', candidates=len(results))


if __name__ == '__main__':
    main()
