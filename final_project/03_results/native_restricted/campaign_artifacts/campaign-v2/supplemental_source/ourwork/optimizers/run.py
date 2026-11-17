"""Run a proposed optimizer through the unchanged frozen baseline evaluator.

Requires the matching baseline's first ten proposals and a completed baseline.
The common package is not patched on disk; a scoped sampler factory supplies
the alternative proposal algorithm to its existing run loop.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from ourwork.baseline.records import atomic_json, fingerprint
from .bridge import EvolutionSampler
from .space import ConditionalSpace
from .nsga2 import PinnedTPESampler


def implementation_identity():
    directory = Path(__file__).parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.glob('*.py'))}
    return {'files': files, 'sha256': fingerprint(files)}


def run(args):
    import optuna
    if optuna.__version__ != '4.2.1':
        raise RuntimeError('Restore pinned Optuna 4.2.1')
    reference = Path(args.baseline_run).resolve()
    baseline_protocol = json.loads((reference / 'protocol.json').read_text())
    summary = json.loads((reference / 'analysis/summary.json').read_text())
    if not summary['budget_exhausted']:
        raise ValueError('Finish the matching baseline attempt budget before optimizer experiments')
    if baseline_protocol['optimizer_seed'] != args.seed:
        raise ValueError('Use the matching baseline seed for shared initialization')
    protocol = dict(baseline_protocol['experiment'])
    if (protocol['budget'], protocol['qa_count'], protocol['startup_trials']) != (30, 20, 10):
        raise ValueError('Expected the frozen 30-attempt, 20-question, 10-startup campaign')
    baseline = optuna.load_study(study_name='baseline', storage=f'sqlite:///{reference / "study.sqlite3"}')
    all_trials = baseline.get_trials(deepcopy=False)
    attempted = [t for t in all_trials if t.state != optuna.trial.TrialState.WAITING]
    if len(attempted) != protocol['budget'] or any(t.state == optuna.trial.TrialState.RUNNING for t in attempted):
        raise ValueError('Baseline SQLite ledger does not contain the completed attempt budget')
    first = all_trials[:10]
    if len(first) != 10 or any(not t.user_attrs.get('concrete_params') for t in first):
        raise ValueError('Shared initialization requires ten recorded concrete baseline proposals')
    initial = [t.user_attrs['concrete_params'] for t in first]
    identity = implementation_identity()
    protocol.update(optimizer_method=args.method, optimizer_implementation_sha256=identity['sha256'],
                    shared_initial_pool_sha256=fingerprint(initial), population_size=5)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    effective = output / 'optimizer_experiment.json'
    if effective.exists() and json.loads(effective.read_text()) != protocol:
        raise ValueError('Existing optimizer experiment differs; never mutate a saved protocol')
    atomic_json(effective, protocol)
    atomic_json(output / 'optimizer_implementation.json', identity)
    atomic_json(output / 'shared_initial_pool.json', {
        'baseline_run': str(reference), 'selection': 'first ten proposals, no outcome-based filtering',
        'params': initial, 'sha256': fingerprint(initial),
        'fitness_policy': 'reevaluate and charge every attempt through the same evaluator',
    })
    # Configure the local alias and device before importing upstream Syftr classes.
    from ourwork.baseline.adapter import configure
    configure(protocol, args.workspace, baseline_protocol['sources'])
    from syftr.studies import SearchSpace
    from ourwork.baseline.spec import search_spec
    codec = ConditionalSpace(SearchSpace(**search_spec(protocol)))
    common_args = argparse.Namespace(
        protocol=str(effective), output=str(output), workspace=args.workspace, seed=args.seed,
        server_metadata=args.server_metadata, preflight=args.preflight,
        capabilities_run=args.capabilities_run, max_hours=args.max_hours, resume=args.resume,
    )
    def sampler_factory(*positional, **kwargs):
        if not positional and not kwargs:
            # Optuna's read-only load_study calls in the existing report builder
            # construct a default sampler even though no proposals are requested.
            return PinnedTPESampler()
        if positional or kwargs.get('seed') != args.seed or kwargs.get('n_startup_trials') != 10:
            raise ValueError('Unexpected baseline sampler construction')
        return EvolutionSampler(codec, initial, args.seed, args.method,
                                protocol['budget'], 5, protocol['resource_reference'])
    from ourwork.baseline.run import run as common_run
    # The hybrid retains a captured original TPESampler for its BO component.
    with patch('optuna.samplers.TPESampler', sampler_factory):
        common_run(common_args)
    study = optuna.load_study(study_name='baseline', storage=f'sqlite:///{output / "study.sqlite3"}')
    atomic_json(output / 'optimizer_generation_log.json', study.user_attrs.get('optimizer_generations', []))
    from ourwork.baseline.run import restore_sampler
    saved_sampler = restore_sampler(output)
    if not isinstance(saved_sampler, EvolutionSampler):
        raise ValueError('Saved sampler is not the declared optimizer')
    atomic_json(output / 'optimizer_state.json', saved_sampler.engine.state_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['nsga2', 'hybrid_motpe', 'adaptive_nsga2'], required=True)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--server-metadata', required=True)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--capabilities-run', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-hours', type=float, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error('--max-hours must be positive')
    run(args)


if __name__ == '__main__':
    main()
