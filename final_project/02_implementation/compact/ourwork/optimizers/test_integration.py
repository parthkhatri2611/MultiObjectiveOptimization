"""Run the actual shared runner against a synthetic worker, never a research result.

This verifies storage, initialization, budget, sampler integration and interruption
recovery without downloading models or claiming synthetic scores as measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings
from unittest.mock import patch
import optuna
from ourwork.baseline.records import atomic_json
from ourwork.baseline.spec import capability_overrides, load_protocol, search_spec
from ourwork.baseline.test_baseline import actual_space_class
from ourwork.optimizers.run import run


def fixture_objectives(params):
    number = int(hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8], 16)
    return (number % 80) / 100, 500 + number % 4000


class SharedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        cls.Native = actual_space_class()
        cls.native_module = types.ModuleType('actual_syftr_space_test')
        cls.native_module.__dict__.update(cls.Native.sample.__globals__)

    def fixture(self, directory):
        root = Path(__file__).resolve().parents[2]
        protocol = load_protocol(root / 'ourwork/experiment_protocols/baseline_f1_gpu1_v1.json')
        native = self.Native(**search_spec(protocol))
        reference = directory / 'baseline'
        reference.mkdir()
        sampler = optuna.samplers.TPESampler(seed=11, n_startup_trials=10,
                                             multivariate=True, constant_liar=True)
        baseline = optuna.create_study(study_name='baseline', sampler=sampler,
                    storage=f'sqlite:///{reference / "study.sqlite3"}', directions=['maximize', 'minimize'])
        for name, params in capability_overrides()[:5]:
            baseline.enqueue_trial({**params, 'rag_embedding_model': protocol['embedding_models'][0]})
        for _ in range(30):
            trial = baseline.ask()
            params = native.sample(trial)
            trial.set_user_attr('concrete_params', params)
            baseline.tell(trial, fixture_objectives(params))
        sources = {'model': {'revision': 'model-pin'}, 'llama': {'revision': 'server-pin'}}
        server = {'model_revision': 'model-pin', 'server_revision': 'server-pin', 'context': 16384}
        identity = {'files': {}, 'sha256': 'test-fixture-only'}
        wrapped = {'experiment': protocol, 'optimizer_seed': 11, 'sources': sources,
                   'source_identity': identity, 'server': server}
        atomic_json(reference / 'protocol.json', wrapped)
        atomic_json(reference / 'analysis/summary.json', {'budget_exhausted': True})
        cap = directory / 'capabilities'
        cp = dict(protocol, stage='capabilities')
        atomic_json(cap / 'protocol.json', dict(wrapped, experiment=cp))
        atomic_json(cap / 'analysis/summary.json', {
            'capability_results': {name: 'COMPLETE' for name, _ in capability_overrides()}})
        atomic_json(cap / 'embedding_checks.json', {name: {'status': 'passed'}
                                                   for name in protocol['embedding_models']})
        atomic_json(directory / 'server.json', server)
        atomic_json(directory / 'preflight.json', {'status': 'passed'})
        study_config = types.SimpleNamespace(search_space=native,
                                             model_dump=lambda **kwargs: {'test_fixture': True})
        environment = {'packages': {'stdout': 'fixture packages'}, 'gpu': {'stdout': ''}}
        return protocol, sources, identity, study_config, environment

    def exercise(self, method, interrupt=False):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol, sources, identity, config, environment = self.fixture(directory)
            fake_syftr = types.ModuleType('syftr')
            fake_syftr.__path__ = []
            fake_studies = types.ModuleType('syftr.studies')
            fake_studies.SearchSpace = self.Native
            args = argparse.Namespace(method=method, baseline_run=str(directory / 'baseline'),
                output=str(directory / method), workspace=str(directory / 'workspace'), seed=11,
                server_metadata=str(directory / 'server.json'), preflight=str(directory / 'preflight.json'),
                capabilities_run=str(directory / 'capabilities'), max_hours=4, resume=False)
            pending_interrupt = [interrupt]
            def worker(job, timeout, root):
                trial_number = int(Path(job['destination']).name.split('-')[0])
                if pending_interrupt[0] and trial_number >= 12:
                    pending_interrupt[0] = False
                    raise KeyboardInterrupt('test interruption')
                quality, cost = fixture_objectives(job['params'])
                return {'status': 'evaluated', 'metrics': {
                    'full_evaluation': True, 'usage_verified': True,
                    'generation_success_rate': 1, 'judge_success_rate': 1,
                    'f1': quality, 'accuracy': quality, 'total_tokens_mean': cost}}
            patches = [
                patch.dict(sys.modules, {'syftr': fake_syftr, 'syftr.studies': fake_studies,
                                         'actual_syftr_space_test': self.native_module}),
                patch('ourwork.baseline.adapter.configure'),
                patch('ourwork.baseline.run.resolve_sources', return_value=sources),
                patch('ourwork.baseline.run.source_identity', return_value=identity),
                patch('ourwork.baseline.run.environment', return_value=environment),
                patch('ourwork.baseline.run.prepare', return_value=(config, None, None)),
                patch('ourwork.baseline.run.question_manifest', return_value=([], {'test_fixture': True})),
                patch('ourwork.baseline.run.grounding_manifest', return_value={'test_fixture': True}),
                patch('ourwork.baseline.run.execute_job', side_effect=worker),
            ]
            from contextlib import ExitStack, redirect_stdout
            import io
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                stack.enter_context(redirect_stdout(io.StringIO()))
                if interrupt:
                    with self.assertRaises(KeyboardInterrupt):
                        run(args)
                    args.resume = True
                run(args)
            study = optuna.load_study(study_name='baseline',
                       storage=f'sqlite:///{directory / method / "study.sqlite3"}')
            self.assertEqual(len(study.trials), 30)
            self.assertTrue(all(t.state.is_finished() for t in study.trials))
            first = [t.user_attrs['concrete_params'] for t in study.trials[:10]]
            baseline = optuna.load_study(study_name='baseline',
                       storage=f'sqlite:///{directory / "baseline/study.sqlite3"}')
            self.assertEqual(first, [t.user_attrs['concrete_params'] for t in baseline.trials[:10]])
            self.assertEqual(sum(t.user_attrs.get('proposal_origin') == 'motpe_injection'
                                 for t in study.trials), 8 if method == 'hybrid_motpe' else 0)
            self.assertEqual(len(study.user_attrs['optimizer_generations']), 4)
            state = json.loads((directory / method / 'optimizer_state.json').read_text())
            self.assertEqual(len(state['history']), 30)
            if interrupt:
                self.assertTrue(any(t.state == optuna.trial.TrialState.FAIL for t in study.trials))

    def test_all_methods_use_the_actual_shared_loop(self):
        for method in ['nsga2', 'hybrid_motpe', 'adaptive_nsga2']:
            with self.subTest(method=method), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.exercise(method)

    def test_shared_loop_recovers_after_interruption(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.exercise('adaptive_nsga2', interrupt=True)


if __name__ == '__main__':
    unittest.main()
