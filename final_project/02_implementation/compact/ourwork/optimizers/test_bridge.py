"""Exercise real Optuna trial storage and sampler checkpoint recovery, without LLMs."""
import pickle
import unittest
import optuna
from optuna.trial import TrialState
from ourwork.baseline.records import fingerprint
from .bridge import EvolutionSampler
from .space import ConditionalSpace
from .test_optimizers import ToySpace, toy_objectives


class BridgeTests(unittest.TestCase):
    def setUp(self):
        import random
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.codec = ConditionalSpace(ToySpace())
        rng = random.Random(3)
        self.initial = [self.codec.materialize(self.codec.random_genome(rng)) for _ in range(10)]

    def evaluate(self, study, fail=False):
        trial = study.ask()
        params = self.codec.space.sample(trial)
        trial.set_user_attr('concrete_params', params)
        if fail:
            study.tell(trial, state=TrialState.FAIL)
        else:
            study.tell(trial, toy_objectives(params))
        return params

    def test_real_trial_callbacks_and_json_logs_for_all_methods(self):
        for method in ['nsga2', 'hybrid_motpe', 'adaptive_nsga2']:
            sampler = EvolutionSampler(self.codec, self.initial, 11, method)
            study = optuna.create_study(sampler=sampler, directions=['maximize', 'minimize'])
            for i in range(30):
                self.evaluate(study, fail=(i == 14))
            self.assertEqual(len(sampler.engine.history), 30)
            self.assertIsNone(sampler.engine.history[14].values)
            self.assertEqual(len(study.user_attrs['optimizer_generations']), 4)
            self.assertEqual(study.user_attrs['optimizer_method'], method)

    def test_lagging_sampler_checkpoint_reconciles_committed_trials(self):
        sampler = EvolutionSampler(self.codec, self.initial, 23, 'nsga2')
        study = optuna.create_study(sampler=sampler, directions=['maximize', 'minimize'])
        for _ in range(10):
            self.evaluate(study)
        stale = pickle.dumps(sampler)
        self.evaluate(study)
        expected = sampler.engine.ask().params
        # Database has 11 completed trials, but saved sampler has only 10.
        restored = pickle.loads(stale)
        study.sampler = restored
        trial = study.ask()
        actual = self.codec.space.sample(trial)
        self.assertEqual(fingerprint(actual), fingerprint(expected))
        self.assertEqual(len(restored.engine.history), 11)

    def test_interruption_before_sampling_still_charges_the_attempt(self):
        sampler = EvolutionSampler(self.codec, self.initial, 37, 'nsga2')
        study = optuna.create_study(sampler=sampler, directions=['maximize', 'minimize'])
        for _ in range(10):
            self.evaluate(study)
        stale = pickle.dumps(sampler)
        interrupted = study.ask()  # Crash before the next sampler save/parameter sampling.
        restored = pickle.loads(stale)
        study.sampler = restored
        study.tell(interrupted.number, state=TrialState.FAIL)
        self.assertEqual(len(restored.engine.history), 11)
        self.assertIsNone(restored.engine.history[-1].values)
        self.evaluate(study)
        self.assertEqual(len(restored.engine.history), 12)


if __name__ == '__main__':
    unittest.main()
