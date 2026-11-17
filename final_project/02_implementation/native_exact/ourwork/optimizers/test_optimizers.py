"""Algorithm and actual conditional-space tests; no synthetic research conclusions."""
import json
import math
import random
import unittest
import warnings
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from ourwork.baseline.test_baseline import actual_space_class
from ourwork.baseline.spec import load_protocol, search_spec
from ourwork.optimizers.space import ConditionalSpace
from ourwork.optimizers.nsga2 import (
    Candidate, EvolutionSearch, crowding_distance, environmental_select, nondominated_sort,
)
from pathlib import Path


class ToySpace:
    def build_distributions(self, params=None):
        d = {'branch': CategoricalDistribution(['a', 'b']),
             'x': IntDistribution(0, 20), 'y': FloatDistribution(0, 1, step=.1)}
        return {k: v for k, v in d.items() if params is None or k in params}

    def sample(self, trial):
        params = {'branch': trial.suggest_categorical('branch', ['a', 'b']),
                  'x': trial.suggest_int('x', 0, 20)}
        if params['branch'] == 'b':
            params['y'] = trial.suggest_float('y', 0, 1, step=.1)
        return params


def toy_objectives(params):
    # Test fixture only, never reported as measured model quality.
    x, y = params['x'] / 20, params.get('y', 0)
    return .1 + .8 * x, 100 + 15000 * x * x + 1000 * y


class SortingTests(unittest.TestCase):
    def test_fronts_with_ties_and_dominated_points(self):
        fronts = nondominated_sort([(.4, 5), (.7, 8), (.3, 9), (.4, 5), (.1, 10)])
        self.assertEqual(set(fronts[0]), {0, 1, 3})
        self.assertEqual(fronts[1], [2])
        self.assertEqual(fronts[2], [4])

    def test_constant_objectives_do_not_make_arbitrary_boundaries(self):
        self.assertEqual(crowding_distance([(.5, 10)] * 3, [0, 1, 2]), {0: 0, 1: 0, 2: 0})

    def test_varying_front_preserves_extremes(self):
        points = [(.1, 1), (.3, 2), (.5, 4), (.9, 9)]
        candidates = [Candidate({'x': i}, {'x': i}, 'test', values=v) for i, v in enumerate(points)]
        survivors = environmental_select(candidates, 2)
        self.assertEqual({c.params['x'] for c in survivors}, {0, 3})

    def test_invalid_or_incomplete_points_are_not_front_points(self):
        with self.assertRaises(ValueError):
            nondominated_sort([(math.nan, 1)])
        c = Candidate({}, {}, 'failed')
        self.assertEqual(environmental_select([c], 5), [])

    def test_sort_matches_independent_bruteforce_for_every_front(self):
        from ourwork.baseline.metrics import pareto_indices
        rng = random.Random(31)
        for _ in range(40):
            values = [(rng.randrange(8) / 8, rng.randrange(20)) for _ in range(25)]
            remaining, expected = list(range(len(values))), []
            while remaining:
                front = {remaining[i] for i in pareto_indices([values[i] for i in remaining])}
                expected.append(front)
                remaining = [i for i in remaining if i not in front]
            self.assertEqual([set(f) for f in nondominated_sort(values)], expected)


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.codec = ConditionalSpace(ToySpace())
        rng = random.Random(9)
        self.initial = [self.codec.materialize(self.codec.random_genome(rng)) for _ in range(10)]

    def run_engine(self, method, budget=30, flat=False):
        search = EvolutionSearch(self.codec, self.initial, 11, method, budget=budget)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            while len(search.history) < search.budget:
                proposal = search.ask()
                search.tell((.5, 1000) if flat else toy_objectives(proposal.params))
        return search

    def test_generational_budget_and_hybrid_injection_slots(self):
        for method in EvolutionSearch.METHODS:
            search = self.run_engine(method)
            self.assertEqual(len(search.history), 30)
            self.assertEqual(len(search.generations), 4)
            self.assertEqual(sum(c.origin == 'shared_initialization' for c in search.history), 10)
            self.assertEqual(sum(c.origin == 'motpe_injection' for c in search.history),
                             8 if method == 'hybrid_motpe' else 0)
            self.assertTrue(all(len(g) <= 5 for g in [search.population]))
            with self.assertRaises(StopIteration):
                search.ask()

    def test_partial_final_generation_uses_no_extra_evaluations(self):
        search = self.run_engine('nsga2', budget=27)
        self.assertEqual([g['offspring_attempts'] for g in search.generations], [5, 5, 5, 2])

    def test_adaptation_changes_only_mutation_control(self):
        fixed = self.run_engine('nsga2', flat=True)
        adaptive = self.run_engine('adaptive_nsga2', flat=True)
        self.assertEqual(fixed.mutation_multiplier, 1)
        self.assertGreater(adaptive.mutation_multiplier, 1)
        self.assertLessEqual(adaptive.mutation_multiplier, 5)
        self.assertTrue(all(0 <= g['conditional_diversity'] <= 1 for g in adaptive.generations))

    def test_failures_are_charged_and_excluded(self):
        search = EvolutionSearch(self.codec, self.initial, 11, budget=15)
        for _ in range(15):
            search.ask()
            search.tell(None, 'test failure')
        self.assertEqual(len(search.history), 15)
        self.assertEqual(search.front(), [])
        self.assertEqual(search.population, [])

    def test_pending_and_nonfinite_values_are_rejected(self):
        search = EvolutionSearch(self.codec, self.initial, 11)
        with self.assertRaises(RuntimeError):
            search.tell((.5, 1))
        search.ask()
        with self.assertRaises(RuntimeError):
            search.ask()
        with self.assertRaises(ValueError):
            search.tell((.5, float('inf')))

    def test_json_checkpoint_preserves_next_proposals_for_all_methods(self):
        for method in EvolutionSearch.METHODS:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
                a = EvolutionSearch(self.codec, self.initial, 23, method)
                for _ in range(18):
                    p = a.ask(); a.tell(toy_objectives(p.params))
                restored = EvolutionSearch.from_state(self.codec, json.loads(json.dumps(a.state_dict(), allow_nan=False)))
                while len(a.history) < 30:
                    p, q = a.ask(), restored.ask()
                    self.assertEqual(p.params, q.params)
                    self.assertEqual(p.origin, q.origin)
                    a.tell(toy_objectives(p.params));restored.tell(toy_objectives(q.params))
                self.assertEqual(a.state_dict(), restored.state_dict())

    def test_pending_checkpoint_keeps_the_proposal(self):
        search = EvolutionSearch(self.codec, self.initial, 11)
        pending = search.ask()
        restored = EvolutionSearch.from_state(self.codec, json.loads(json.dumps(search.state_dict())))
        self.assertEqual(restored.pending.params, pending.params)
        restored.tell(None, 'interrupted')
        self.assertEqual(len(restored.history), 1)

    def test_archive_hypervolume_never_decreases(self):
        search = self.run_engine('nsga2')
        hvs = [g['hypervolume'] for g in search.generations]
        self.assertEqual(hvs, sorted(hvs))


class NativeSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        protocol = load_protocol(root / 'ourwork/baseline/configs/baseline_f1_tokens.json')
        cls.codec = ConditionalSpace(actual_space_class()(**search_spec(protocol)))

    def test_native_conditional_crossover_and_mutation_stay_valid(self):
        rng = random.Random(37)
        a, b = self.codec.random_genome(rng), self.codec.random_genome(rng)
        modes = set()
        for _ in range(300):
            a, probability = self.codec.offspring(a, b, rng, multiplier=3)
            params = self.codec.materialize(a)
            imported = self.codec.import_params(params, rng)
            self.assertEqual(self.codec.materialize(imported), params)
            self.assertLessEqual(probability, .5)
            modes.add(params['rag_mode'])
            if rng.random() < .25:
                b = self.codec.random_genome(rng)
        self.assertEqual(len(modes), 5)

    def test_crossover_inherits_each_module_from_one_parent(self):
        rng = random.Random(12)
        a, b = self.codec.random_genome(rng), self.codec.random_genome(rng)
        for _ in range(50):
            child = self.codec.crossover(a, b, rng, 1)
            for block in {self.codec.block(k) for k in self.codec.distributions}:
                keys = [k for k in self.codec.distributions if self.codec.block(k) == block]
                self.assertTrue(all(child[k] == a[k] for k in keys)
                                or all(child[k] == b[k] for k in keys))

    def test_hybrid_accepts_actual_syftr_conditional_observations(self):
        rng = random.Random(11)
        initial = [self.codec.materialize(self.codec.random_genome(rng)) for _ in range(10)]
        search = EvolutionSearch(self.codec, initial, 11, 'hybrid_motpe', budget=15)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
            for i in range(15):
                proposal = search.ask()
                self.assertEqual(self.codec.materialize(proposal.genome), proposal.params)
                search.tell((i / 20, 1000 + i * 5))
        self.assertEqual(sum(c.origin == 'motpe_injection' for c in search.history), 2)


if __name__ == '__main__':
    unittest.main()
