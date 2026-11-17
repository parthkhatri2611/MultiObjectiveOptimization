"""Conditional genetic representation backed by the actual Syftr SearchSpace."""
import math
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.trial import FixedTrial
from ourwork.baseline.records import fingerprint


class ConditionalSpace:
    def __init__(self, space):
        self.space = space
        self.distributions = space.build_distributions()
        self.mutable = {k for k, d in self.distributions.items() if not d.single()}

    @staticmethod
    def draw(distribution, rng, previous=None, exclude_previous=False):
        d = distribution
        if isinstance(d, CategoricalDistribution):
            choices = list(d.choices)
            if exclude_previous and len(choices) > 1:
                choices = [v for v in choices if v != previous]
            return rng.choice(choices)
        if d.single():
            return d.low
        step = getattr(d, 'step', None)
        if step is not None:
            count = int(round((d.high - d.low) / step)) + 1
            old = int(round((previous - d.low) / step)) if previous is not None else -1
            if exclude_previous and count > 1:
                index = rng.randrange(count - 1)
                if index >= old:
                    index += 1
            elif getattr(d, 'log', False):
                raw = math.exp(rng.uniform(math.log(d.low), math.log(d.high)))
                index = min(count - 1, max(0, int(round((raw - d.low) / step))))
            else:
                index = rng.randrange(count)
            value = d.low + index * step
            return int(value) if isinstance(d, IntDistribution) else round(float(value), 12)
        return (math.exp(rng.uniform(math.log(d.low), math.log(d.high)))
                if d.log else rng.uniform(d.low, d.high))

    def random_genome(self, rng):
        return {k: self.draw(d, rng) for k, d in self.distributions.items()}

    def materialize(self, genome):
        # The native sampler activates/deactivates modules and their parameters.
        return self.space.sample(FixedTrial(genome))

    def import_params(self, params, rng):
        unknown = set(params) - self.distributions.keys()
        if unknown:
            raise ValueError(f'Unknown initial parameters: {sorted(unknown)}')
        genome = self.random_genome(rng)
        for key, value in params.items():
            d = self.distributions[key]
            if not d._contains(d.to_internal_repr(value)):
                raise ValueError(f'Initial parameter outside distribution: {key}={value}')
            genome[key] = value
        actual = self.materialize(genome)
        if fingerprint(actual) != fingerprint(params):
            raise ValueError('Initial parameters do not match the active native search space')
        return genome

    @staticmethod
    def block(key):
        if key == 'rag_mode':
            return 'workflow'
        for prefix in ['few_shot_', 'hyde_', 'reranker_', 'additional_context_', 'splitter_']:
            if key.startswith(prefix):
                return prefix
        if key.startswith('rag_'):
            return 'retrieval'
        return key

    def crossover(self, left, right, rng, crossover_probability=.9):
        if rng.random() < crossover_probability:
            choices = {block: rng.random() < .5 for block in sorted({self.block(k) for k in self.distributions})}
            return {k: (left[k] if choices[self.block(k)] else right[k]) for k in self.distributions}
        return dict(left)

    def offspring(self, left, right, rng, multiplier=1.0, crossover_probability=.9):
        genome = self.crossover(left, right, rng, crossover_probability)
        active = sorted(set(self.materialize(genome)) & self.mutable)
        probability = min(.5, multiplier / max(1, len(active)))
        for key in active:
            if rng.random() < probability:
                genome[key] = self.draw(self.distributions[key], rng, genome[key], True)
        # Prevent a no-op parent copy without evaluating/ranking alternative children.
        # Global duplicates remain charged failures in the experiment runner.
        parents = {fingerprint(self.materialize(p)) for p in [left, right]}
        if active and fingerprint(self.materialize(genome)) in parents:
            key = rng.choice(active)
            genome[key] = self.draw(self.distributions[key], rng, genome[key], True)
        return genome, probability

    def distance(self, left, right):
        keys = (set(left) | set(right)) & self.mutable
        if not keys:
            return 0.0
        total = 0.0
        for key in keys:
            if key not in left or key not in right:
                total += 1.0
                continue
            d = self.distributions[key]
            if isinstance(d, CategoricalDistribution):
                total += float(left[key] != right[key])
            elif d.log:
                total += abs(math.log(left[key]) - math.log(right[key])) / (math.log(d.high) - math.log(d.low))
            else:
                total += abs(left[key] - right[key]) / (d.high - d.low)
        return total / len(keys)

    def active_distributions(self, params):
        return self.space.build_distributions(params)

