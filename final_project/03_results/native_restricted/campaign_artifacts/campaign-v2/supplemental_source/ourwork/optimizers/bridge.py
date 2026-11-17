"""Optuna storage adapter, not an Optuna evolutionary sampler implementation.

The baseline's existing run/evaluate/accounting loop is reused byte-for-byte.
Only proposal generation and feedback are delegated to our EvolutionSearch.
"""
import time
from optuna.samplers import BaseSampler
from optuna.trial import TrialState
from ourwork.baseline.records import fingerprint
from .nsga2 import EvolutionSearch


class EvolutionSampler(BaseSampler):
    def __init__(self, codec, initial_params, seed, method, budget=30, population_size=5,
                 resource_reference=50000):
        self.codec, self.initial_params, self.seed, self.method = codec, initial_params, seed, method
        self.budget, self.population_size, self.resource_reference = budget, population_size, resource_reference
        self.engine = self._new_engine()

    def _new_engine(self):
        return EvolutionSearch(self.codec, self.initial_params, self.seed, self.method,
                               self.budget, self.population_size, self.resource_reference)

    def _check(self, proposal, trial):
        params = trial.user_attrs.get('concrete_params')
        if params is not None and fingerprint(params) != proposal.identity:
            raise ValueError('Saved trial does not match deterministic optimizer replay')

    def _rebuild_before(self, study, number):
        # Recover a sampler checkpoint lagging a committed SQLite trial. No model
        # evaluations are replayed, and no outcomes are fabricated.
        self.engine = self._new_engine()
        past = [t for t in study.get_trials(deepcopy=False) if t.number < number]
        for expected_number, old in enumerate(sorted(past, key=lambda t: t.number)):
            if old.number != expected_number or old.state not in {TrialState.COMPLETE, TrialState.FAIL}:
                raise ValueError('Cannot reconcile incomplete or nonsequential optimizer ledger')
            proposal = self.engine.ask()
            self._check(proposal, old)
            self.engine.tell(old.values if old.state == TrialState.COMPLETE else None,
                             old.user_attrs.get('failure_reason', 'replayed_failure'))
        if len(self.engine.history) != number:
            raise ValueError('Optimizer and trial ledger disagree')

    def before_trial(self, study, trial):
        start = time.monotonic()
        if len(self.engine.history) != trial.number or self.engine.pending is not None:
            self._rebuild_before(study, trial.number)
        proposal = self.engine.ask()
        for key, value in trial.system_attrs.get('fixed_params', {}).items():
            if key in proposal.params and proposal.params[key] != value:
                raise ValueError('Baseline initialization differs from the shared initialization pool')
        metadata = {'optimizer_method': self.method, 'proposal_origin': proposal.origin,
                    'proposal_seconds': time.monotonic() - start,
                    'mutation_probability': proposal.mutation_probability,
                    'mutation_multiplier': self.engine.mutation_multiplier,
                    'evolution_generation': len(self.engine.generations)}
        for key, value in metadata.items():
            study._storage.set_trial_user_attr(trial._trial_id, key, value)

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        if self.engine.pending is None:
            raise RuntimeError('No proposed genetic candidate for the active trial')
        value = self.engine.pending.genome[param_name]
        if not param_distribution._contains(param_distribution.to_internal_repr(value)):
            raise ValueError(f'Proposed value outside native distribution: {param_name}')
        return value

    def after_trial(self, study, trial, state, values):
        if len(self.engine.history) != trial.number or self.engine.pending is None:
            self._rebuild_before(study, trial.number)
            self.engine.ask()
        latest = study._storage.get_trial(trial._trial_id)
        self._check(self.engine.pending, latest)
        self.engine.tell(values if state == TrialState.COMPLETE else None,
                         latest.user_attrs.get('failure_reason', 'failed_or_interrupted'))
        study.set_user_attr('optimizer_method', self.method)
        study.set_user_attr('optimizer_generations', self.engine.generations)
        study.set_user_attr('optimizer_population', [c.identity for c in self.engine.population])

