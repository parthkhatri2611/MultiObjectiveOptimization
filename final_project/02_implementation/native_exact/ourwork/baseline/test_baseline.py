"""Run with: python -m unittest ourwork.baseline.test_baseline -v.

Search-space tests compile the actual dependency-light schema classes from studies.py.
They do not claim to test LlamaIndex or inference; full integration runs on Kaggle.
"""
import ast
from abc import ABC, abstractmethod
import json
from pathlib import Path
import tempfile
import threading
import typing as T
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
from .metrics import answer_scores, hypervolume, igd, pareto_indices
from .records import RunStore, append_jsonl, read_jsonl, repair_trailing_record
from .spec import PAPER_MODES, load_protocol, search_spec


def actual_space_class():
    import optuna
    from pydantic import BaseModel, ConfigDict, Field
    from optuna import distributions
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / 'syftr/studies.py').read_text())
    names = ['SearchSpaceMixin', 'Splitter', 'TopK', 'Hybrid', 'LLMConfig', 'QueryDecomposition',
             'FusionMode', 'Retriever', 'FewShotRetriever', 'Reranker', 'Hyde', 'AdditionalContext',
             'ReactRAGAgent', 'CritiqueRAGAgent', 'SubQuestionRAGAgent', 'LATSRagAgent', 'CoARagAgent',
             'SearchSpace', 'get_dist_cardinality']
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign)) and
              any(isinstance(t, ast.Name) and t.id in {'PARAMETERS', 'RAG_MODES', 'TEMPLATE_NAMES'}
                  for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    nodes.sort(key=lambda n: n.lineno)
    namespace = {'T': T, 'ABC': ABC, 'abstractmethod': abstractmethod, 'BaseModel': BaseModel,
                 'ConfigDict': ConfigDict, 'Field': Field, 'Trial': optuna.Trial, 'NDIGITS': 5,
                 'ParamDict': T.Dict[str, str | int | float | bool], 'LLM_NAMES': ['rm-qwen25-7b-q4'],
                 'DEFAULT_EMBEDDING_MODELS': ['BAAI/bge-small-en-v1.5'], '__name__': 'actual_syftr_space_test'}
    namespace.update({name: getattr(distributions, name) for name in
                      ['BaseDistribution', 'CategoricalDistribution', 'FloatDistribution', 'IntDistribution',
                       'DiscreteUniformDistribution', 'LogUniformDistribution', 'UniformDistribution']})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / 'syftr/studies.py'), 'exec'), namespace)
    return namespace['SearchSpace']


class MetricsTest(unittest.TestCase):
    def test_multiset_f1(self):
        self.assertAlmostEqual(answer_scores('cat cat', 'cat cat')['f1'], 1)
        self.assertAlmostEqual(answer_scores('cat cat', 'cat')['f1'], 2 / 3)

    def test_yes_no(self):
        self.assertEqual(answer_scores('yes indeed', 'yes')['f1'], 0)
        self.assertEqual(answer_scores('YES', 'yes')['exact_match'], 1)

    def test_normalization(self):
        self.assertEqual(answer_scores(' The Eiffel Tower! ', 'eiffel tower')['exact_match'], 1)

    def test_empty(self):
        self.assertEqual(answer_scores('', 'Paris')['f1'], 0)

    def test_front_and_hv(self):
        points = [(.5, 10), (.8, 20), (.3, 30)]
        self.assertEqual(pareto_indices(points), [0, 1])
        self.assertAlmostEqual(hypervolume(points, 100), .69)
        self.assertEqual(hypervolume([], 100), 0)

    def test_hv_monotonicity_and_ties(self):
        self.assertEqual(hypervolume([(.5, 10), (.5, 10)], 100), .45)
        self.assertGreater(hypervolume([(.5, 10), (.8, 20)], 100), hypervolume([(.5, 10)], 100))

    def test_hv_outside_reference(self):
        self.assertEqual(hypervolume([(.8, 101)], 100), 0)

    def test_igd(self):
        self.assertEqual(igd([(.5, 10)], [(.5, 10)], 100), 0)
        self.assertIsNone(igd([], [(.5, 10)], 100))


class StorageTest(unittest.TestCase):
    def test_partial_and_unverified_candidates_cannot_rank(self):
        from .run import eligible
        p = {'minimum_success_rate': .95, 'objective_quality': 'accuracy'}
        m = {'full_evaluation': True, 'usage_verified': True,
             'generation_success_rate': 1, 'judge_success_rate': 1}
        self.assertTrue(eligible(m, p))
        self.assertFalse(eligible({**m, 'full_evaluation': False}, p))
        self.assertFalse(eligible({**m, 'usage_verified': False}, p))
        self.assertFalse(eligible({**m, 'judge_success_rate': .5}, p))

    def test_findings_exclude_failed_trials(self):
        import optuna
        from .analyze import analyze
        from .records import atomic_json
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            p = {'stage': 'test_fixture', 'dataset': 'synthetic_fixture', 'partition': 'test', 'budget': 3,
                 'objective_quality': 'accuracy', 'objective_resource': 'total_tokens_mean', 'resource_reference': 100}
            atomic_json(path / 'protocol.json', {'experiment': p})
            study = optuna.create_study(storage=f'sqlite:///{path / "study.sqlite3"}',
                                        study_name='baseline', directions=['maximize', 'minimize'])
            study.add_trial(optuna.trial.create_trial(values=[.5, 10]))
            study.add_trial(optuna.trial.create_trial(values=[.8, 20]))
            study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
            result = analyze(path)
            self.assertEqual(result['failed'], 1)
            self.assertEqual(result['pareto_count'], 2)
            self.assertAlmostEqual(result['hypervolume'], .69)
            self.assertTrue((path / 'analysis/FINDINGS.md').exists())

    def test_resume_identity(self):
        with tempfile.TemporaryDirectory() as d:
            RunStore(d, {'seed': 1})
            RunStore(d, {'seed': 1}, resume=True)
            with self.assertRaises(ValueError):
                RunStore(d, {'seed': 2}, resume=True)
            with self.assertRaises(FileExistsError):
                RunStore(d, {'seed': 1})

    def test_torn_journal(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'questions.jsonl'
            append_jsonl(p, {'complete': 1})
            with p.open('ab') as f:
                f.write(b'{"partial":')
            self.assertEqual(read_jsonl(p), [{'complete': 1}])
            repair_trailing_record(p)
            append_jsonl(p, {'complete': 2})
            self.assertEqual(read_jsonl(p), [{'complete': 1}, {'complete': 2}])

    def test_earlier_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.jsonl'
            p.write_text('bad\n{"valid":1}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_jsonl(p)

    def test_sampler_restart_same_proposal(self):
        import optuna
        from .run import save_sampler, restore_sampler
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            db = f'sqlite:///{path / "study.db"}'
            sampler = optuna.samplers.TPESampler(seed=11, n_startup_trials=3, multivariate=True)
            study = optuna.create_study(storage=db, sampler=sampler, study_name='test', directions=['maximize', 'minimize'])
            for _ in range(7):
                t = study.ask()
                x = t.suggest_int('x', 1, 10)
                study.tell(t, [x / 10, x])
            save_sampler(path, sampler)
            first = study.ask()
            expected = first.suggest_int('x', 1, 10)
            # Restore on a separate identical history, with the last trial still unvalued.
            restored = optuna.load_study(study_name='test', storage=db, sampler=restore_sampler(path))
            other = restored.ask()
            observed = other.suggest_int('x', 1, 10)
            self.assertEqual(expected, observed)


class SpaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol_path = Path(__file__).parent / 'configs/baseline.json'
        cls.protocol = load_protocol(cls.protocol_path)
        cls.Space = actual_space_class()

    def test_original_hierarchy_and_distributions(self):
        import optuna
        space = self.Space(**search_spec(self.protocol))
        self.assertEqual(space.rag_modes, PAPER_MODES)
        self.assertEqual(len(space.rag_retriever.embedding_models), 11)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3), directions=['maximize', 'minimize'])
        seen = set()
        for _ in range(200):
            t = study.ask()
            params = space.sample(t)
            seen.add(params['rag_mode'])
            self.assertIn(params['splitter_method'], ['sentence', 'token', 'recursive'])
            if params['rag_method'] == 'sparse':
                self.assertNotIn('rag_embedding_model', params)
            if not params['reranker_enabled']:
                self.assertNotIn('reranker_llm_name', params)
            for name, distribution in t.distributions.items():
                self.assertTrue(distribution._contains(distribution.to_internal_repr(t.params[name])))
            study.tell(t, [.5, 100])
        self.assertEqual(seen, set(PAPER_MODES))

    def test_all_profiles_validate(self):
        for p in (Path(__file__).parent / 'configs').glob('*.json'):
            load_protocol(p)

    def test_prevent_holdout_search(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'protocol.json'
            protocol = {**self.protocol, 'partition': 'holdout'}
            p.write_text(json.dumps(protocol))
            with self.assertRaises(ValueError):
                load_protocol(p)


class ProxyTest(unittest.TestCase):
    def test_actual_usage_roles_and_raw_records(self):
        from .proxy import CallProxy

        class Endpoint(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                raw = json.dumps({'choices': [{'message': {'content': 'Paris'}}],
                                  'usage': {'prompt_tokens': 13, 'completion_tokens': 2}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Endpoint)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'calls.jsonl'
                with CallProxy(f'http://127.0.0.1:{upstream.server_port}', path, port=0) as proxy:
                    proxy.phase(candidate_id='abc', qa_id='q1', role='generation')
                    request = urllib.request.Request(f'http://127.0.0.1:{proxy.server.server_port}/v1/chat/completions',
                                                     data=b'{"messages":[],"stream":false}',
                                                     headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request) as response:
                        json.load(response)
                saved = read_jsonl(path)
                self.assertEqual(saved[0]['usage']['prompt_tokens'], 13)
                self.assertTrue(saved[0]['usage_verified'])
                self.assertEqual(saved[0]['role'], 'generation')
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
