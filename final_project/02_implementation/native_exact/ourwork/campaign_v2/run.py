"""One arm through the frozen common runner; only proposal selection is substituted."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from .common import atomic, implementation_identity, read, checked_source_identity
from .snapshots import snapshot


class CampaignPause(BaseException):
    pass


def run_arm(args):
    import optuna
    from ourwork.baseline.spec import load_protocol, search_spec
    from ourwork.baseline.adapter import configure
    from ourwork.baseline.records import fingerprint
    from ourwork.optimizers.nsga2 import PinnedTPESampler
    from ourwork.optimizers.bridge import EvolutionSampler
    from ourwork.optimizers.space import ConditionalSpace
    import ourwork.baseline.run as common
    import ourwork.baseline.analyze as reporting
    if optuna.__version__ != '4.2.1':
        raise ValueError('Pinned Optuna 4.2.1 is required')
    profile = load_protocol(args.protocol)
    campaign = Path(args.campaign)
    if (campaign / 'STOP_REQUESTED').exists():
        return False
    certificate = read(campaign / 'implementation.json')
    if certificate != implementation_identity():
        raise ValueError('Campaign/optimizer implementation changed after installation')
    cap = read(Path(args.capabilities_run) / 'protocol.json')
    sources = cap['sources']
    actual_identity = common.source_identity(Path(__file__).resolve().parents[2])
    compatible_identity = checked_source_identity(actual_identity, cap['source_identity'])
    atomic(campaign / ('actual_common_source_identity_' + args.method + '.json'), actual_identity)
    protocol = dict(profile, optimizer_method=args.method,
                    optimizer_implementation_sha256=certificate['sha256'])
    initial = None
    if args.method != 'syftr_motpe':
        baseline = Path(args.baseline_run)
        bp = read(baseline / 'protocol.json')
        expected_baseline = dict(profile, optimizer_method='syftr_motpe',
                                 optimizer_implementation_sha256=certificate['sha256'])
        if bp['experiment'] != expected_baseline or bp['optimizer_seed'] != args.seed:
            raise ValueError('Reference baseline differs from this campaign or seed')
        study = optuna.load_study(study_name='baseline', storage=f'sqlite:///{baseline / "study.sqlite3"}')
        trials = [t for t in study.trials if t.state != optuna.trial.TrialState.WAITING]
        if not profile['startup_trials'] <= len(trials) <= profile['budget'] or any(not t.state.is_finished() for t in trials):
            raise ValueError('Reference baseline must contain the finished shared initialization pool')
        initial = [t.user_attrs.get('concrete_params') for t in trials[:profile['startup_trials']]]
        if any(not p for p in initial):
            raise ValueError('Baseline initial proposals are missing; do not invent configurations')
        protocol.update(shared_initial_pool_sha256=fingerprint(initial), population_size=5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    effective = output / 'selected_protocol.json'
    if effective.exists() and read(effective) != protocol:
        raise ValueError('Saved arm protocol differs; never edit a resumed experiment')
    atomic(effective, protocol)
    codec = None
    if initial is not None:
        configure(profile, args.workspace, sources)
        from syftr.studies import SearchSpace
        codec = ConditionalSpace(SearchSpace(**search_spec(profile)))
        atomic(output / 'shared_initial_pool.json', {
            'params': initial, 'sha256': fingerprint(initial),
            'policy': 'First baseline proposals by order; all re-evaluated and charged; no fitness copied.'})
    original_analyze = reporting.analyze

    def checkpoint_and_analyze(path):
        result = original_analyze(path)
        snapshot(campaign, args.export_dir)
        target = getattr(args, 'target_attempts', None)
        if ((campaign / 'STOP_REQUESTED').exists() or
                (target is not None and target < profile['budget'] and result['attempts'] >= target)):
            raise CampaignPause('Paused at a completed candidate boundary')
        return result

    def sampler_factory(*positional, **kwargs):
        if not positional and not kwargs:
            return PinnedTPESampler()
        if positional or kwargs.get('seed') != args.seed or kwargs.get('n_startup_trials') != profile['startup_trials']:
            raise ValueError('Unexpected common sampler construction')
        return EvolutionSampler(codec, initial, args.seed, args.method, profile['budget'], 5,
                                profile['resource_reference'])

    common_args = argparse.Namespace(protocol=str(effective), output=str(output), workspace=args.workspace,
        seed=args.seed, max_hours=args.max_hours, resume=(output / 'protocol.json').exists(),
        capabilities_run=args.capabilities_run, server_metadata=args.server_metadata, preflight=args.preflight)
    paused = False
    try:
        with ExitStack() as stack:
            # Use already-audited revisions instead of re-resolving HF HEAD on every arm.
            stack.enter_context(patch.object(common, 'resolve_sources', return_value=sources))
            stack.enter_context(patch.object(common, 'source_identity', return_value=compatible_identity))
            stack.enter_context(patch.object(reporting, 'analyze', checkpoint_and_analyze))
            if initial is not None:
                stack.enter_context(patch('optuna.samplers.TPESampler', sampler_factory))
            common.run(common_args)
    except CampaignPause:
        paused = True
    finally:
        if (output / 'study.sqlite3').exists():
            original_analyze(output)
            if initial is not None and (output / 'sampler_checkpoint.json').exists():
                sampler = common.restore_sampler(output)
                atomic(output / 'optimizer_state.json', sampler.engine.state_dict())
                atomic(output / 'optimizer_generation_log.json', sampler.engine.generations)
        snapshot(campaign, args.export_dir)
    return not paused and read(output / 'analysis/summary.json')['budget_exhausted']


def main():
    from .common import METHODS
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    for name in ['protocol', 'campaign', 'output', 'workspace', 'capabilities-run',
                 'server-metadata', 'preflight', 'export-dir']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--baseline-run')
    parser.add_argument('--target-attempts', type=int, help='Planned intermediate checkpoint; declared total budget is unchanged')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--max-hours', type=float, required=True)
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error('Hours must be positive')
    raise SystemExit(0 if run_arm(args) else 10)


if __name__ == '__main__':
    main()
