"""Generational NSGA-II, adaptive mutation, and budgeted MOTPE injections.

Selection/variation/adaptation are implemented here, not delegated to an NSGA sampler.
Only the hybrid's explicitly identified BO component uses pinned Optuna MOTPE.
"""
from dataclasses import asdict, dataclass
import math
import random
from optuna.samplers import TPESampler as PinnedTPESampler
from ourwork.baseline.metrics import hypervolume, pareto_indices
from ourwork.baseline.records import fingerprint


@dataclass
class Candidate:
    genome: dict
    params: dict
    origin: str
    mutation_probability: float | None = None
    values: tuple | None = None
    failure_reason: str | None = None

    @property
    def identity(self):
        return fingerprint(self.params)


def dominates(left, right):
    return (left[0] >= right[0] and left[1] <= right[1]
            and (left[0] > right[0] or left[1] < right[1]))


def nondominated_sort(values):
    if any(len(v) != 2 or not all(math.isfinite(x) for x in v) for v in values):
        raise ValueError('Sorting requires finite complete objective pairs')
    defeats = [[] for _ in values]
    count = [0] * len(values)
    for i, left in enumerate(values):
        for j in range(i + 1, len(values)):
            if dominates(left, values[j]):
                defeats[i].append(j)
                count[j] += 1
            elif dominates(values[j], left):
                defeats[j].append(i)
                count[i] += 1
    front = [i for i, c in enumerate(count) if c == 0]
    fronts = []
    while front:
        fronts.append(front)
        following = []
        for i in front:
            for j in defeats[i]:
                count[j] -= 1
                if count[j] == 0:
                    following.append(j)
        front = following
    return fronts


def crowding_distance(values, front):
    result = {i: 0.0 for i in front}
    for axis in range(2):
        ordered = sorted(front, key=lambda i: (values[i][axis], i))
        if not ordered:
            continue
        span = values[ordered[-1]][axis] - values[ordered[0]][axis]
        if span == 0:
            continue  # Constant objectives must not create arbitrary infinite boundaries.
        result[ordered[0]] = result[ordered[-1]] = math.inf
        for j in range(1, len(ordered) - 1):
            result[ordered[j]] += (values[ordered[j + 1]][axis] - values[ordered[j - 1]][axis]) / span
    return result


def environmental_select(candidates, size):
    valid = [c for c in candidates if c.values is not None]
    values = [c.values for c in valid]
    selected = []
    for front in nondominated_sort(values):
        if len(selected) + len(front) <= size:
            selected.extend(valid[i] for i in front)
        else:
            distances = crowding_distance(values, front)
            ordered = sorted(front, key=lambda i: (-distances[i], valid[i].identity))
            selected.extend(valid[i] for i in ordered[:size - len(selected)])
            break
    return selected


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


class EvolutionSearch:
    METHODS = {'nsga2', 'hybrid_motpe', 'adaptive_nsga2'}

    def __init__(self, codec, initial_params, seed, method='nsga2', budget=30,
                 population_size=5, resource_reference=50000):
        if method not in self.METHODS or population_size < 2:
            raise ValueError('Unknown method or invalid population size')
        if len(initial_params) < population_size or budget < len(initial_params):
            raise ValueError('Need a charged initial pool at least as large as the population')
        self.codec = codec
        self.initial_params = [dict(p) for p in initial_params]
        self.seed, self.method, self.budget = seed, method, budget
        self.population_size, self.resource_reference = population_size, resource_reference
        self.rng = random.Random(seed)
        self.bo_rng = random.Random(seed ^ 0x5F3759DF)
        self.history, self.population, self.children, self.generations = [], [], [], []
        self.pending = None
        self.mutation_multiplier = 1.0
        self.stagnation = 0
        self.previous_hv = 0.0

    def _tournament(self):
        values = [p.values for p in self.population]
        ranks, distances = {}, {}
        for rank, front in enumerate(nondominated_sort(values)):
            ranks.update({i: rank for i in front})
            distances.update(crowding_distance(values, front))
        i, j = self.rng.randrange(len(values)), self.rng.randrange(len(values))
        a, b = (ranks[i], -distances[i]), (ranks[j], -distances[j])
        winner = i if a < b else j if b < a else self.rng.choice([i, j])
        return self.population[winner]

    def _bayesian_proposal(self):
        import optuna
        sampler = PinnedTPESampler(seed=self.bo_rng.randrange(2**31),
                                   n_startup_trials=10, multivariate=True,
                                   constant_liar=True)
        study = optuna.create_study(sampler=sampler, directions=['maximize', 'minimize'])
        for old in self.history:
            if old.values is not None:
                trial = optuna.trial.create_trial(
                    params=old.params, distributions=self.codec.active_distributions(old.params),
                    values=old.values)
                study.add_trial(trial)
        trial = study.ask()
        params = self.codec.space.sample(trial)
        return Candidate(self.codec.import_params(params, self.rng), params, 'motpe_injection')

    def ask(self):
        if self.pending is not None:
            raise RuntimeError('Tell the pending evaluation before asking again')
        if len(self.history) >= self.budget:
            raise StopIteration('Attempt budget exhausted')
        if len(self.history) < len(self.initial_params):
            params = self.initial_params[len(self.history)]
            candidate = Candidate(self.codec.import_params(params, self.rng), dict(params), 'shared_initialization')
        elif (self.method == 'hybrid_motpe'
              and len(self.children) >= self.population_size - 2):
            # Exactly the last 2 of 5 offspring slots use BO; no free evaluations.
            candidate = self._bayesian_proposal()
        elif not self.population:
            genome = self.codec.random_genome(self.rng)
            candidate = Candidate(genome, self.codec.materialize(genome), 'empty_population_random')
        else:
            left, right = self._tournament(), self._tournament()
            genome, probability = self.codec.offspring(
                left.genome, right.genome, self.rng, self.mutation_multiplier)
            candidate = Candidate(genome, self.codec.materialize(genome), 'genetic', probability)
        self.pending = candidate
        return candidate

    def tell(self, values, failure_reason=None):
        if self.pending is None:
            raise RuntimeError('No pending evaluation')
        if values is not None:
            if (len(values) != 2 or not all(math.isfinite(v) for v in values)
                    or not 0 <= values[0] <= 1 or values[1] < 0):
                raise ValueError('Invalid measured objective values')
            self.pending.values = tuple(float(v) for v in values)
        else:
            self.pending.failure_reason = failure_reason or 'ineligible_or_failed'
        self.history.append(self.pending)
        if len(self.history) == len(self.initial_params):
            self.population = environmental_select(self.history, self.population_size)
            self.previous_hv = hypervolume([p.values for p in self.history if p.values is not None],
                                           self.resource_reference)
        elif len(self.history) > len(self.initial_params):
            self.children.append(self.pending)
            if len(self.children) == self.population_size or len(self.history) == self.budget:
                self._finish_generation()
        self.pending = None

    def _finish_generation(self):
        attempts = len(self.children)
        self.population = environmental_select(self.population + self.children, self.population_size)
        valid = [p.values for p in self.history if p.values is not None]
        hv = hypervolume(valid, self.resource_reference)
        pairs = [(a, b) for i, a in enumerate(self.population) for b in self.population[i + 1:]]
        diversity = (sum(self.codec.distance(a.params, b.params) for a, b in pairs) / len(pairs)
                     if pairs else 0.0)
        gain = hv - self.previous_hv
        self.stagnation = self.stagnation + 1 if gain <= 1e-6 else 0
        previous_multiplier = self.mutation_multiplier
        if self.method == 'adaptive_nsga2':
            if self.stagnation >= 1 or diversity < .15:
                self.mutation_multiplier = min(5.0, self.mutation_multiplier * 1.5)
            else:
                self.mutation_multiplier = max(1.0, self.mutation_multiplier / 1.2)
        self.generations.append({
            'generation': len(self.generations) + 1, 'offspring_attempts': attempts,
            'successful_offspring': sum(c.values is not None for c in self.children),
            'hypervolume': hv, 'hypervolume_gain': gain, 'conditional_diversity': diversity,
            'stagnation_generations': self.stagnation,
            'mutation_multiplier_used': previous_multiplier,
            'mutation_multiplier_next': self.mutation_multiplier,
        })
        self.previous_hv = hv
        self.children = []

    def front(self):
        valid = [p for p in self.history if p.values is not None]
        return [valid[i] for i in pareto_indices([p.values for p in valid])]

    def state_dict(self):
        # JSON-safe state; no executable pickle and no infinite crowding values.
        positions = {id(c): i for i, c in enumerate(self.history)}
        return {
            'version': 1, 'method': self.method, 'seed': self.seed, 'budget': self.budget,
            'population_size': self.population_size, 'resource_reference': self.resource_reference,
            'initial_params': self.initial_params,
            'distribution_hash': fingerprint({k: str(v) for k, v in self.codec.distributions.items()}),
            'history': [asdict(c) for c in self.history],
            'population_indices': [positions[id(c)] for c in self.population],
            'children_indices': [positions[id(c)] for c in self.children],
            'pending': asdict(self.pending) if self.pending else None,
            'generations': self.generations, 'rng_state': self.rng.getstate(),
            'bo_rng_state': self.bo_rng.getstate(), 'mutation_multiplier': self.mutation_multiplier,
            'stagnation': self.stagnation, 'previous_hv': self.previous_hv,
        }

    @classmethod
    def from_state(cls, codec, state):
        if state['version'] != 1:
            raise ValueError('Unknown optimizer state version')
        expected = fingerprint({k: str(v) for k, v in codec.distributions.items()})
        if expected != state['distribution_hash']:
            raise ValueError('Search distributions changed')
        result = cls(codec, state['initial_params'], state['seed'], state['method'],
                     state['budget'], state['population_size'], state['resource_reference'])
        def candidate(item):
            item = dict(item)
            if item['values'] is not None:
                item['values'] = tuple(item['values'])
            return Candidate(**item)
        result.history = [candidate(c) for c in state['history']]
        result.population = [result.history[i] for i in state['population_indices']]
        result.children = [result.history[i] for i in state['children_indices']]
        result.pending = candidate(state['pending']) if state['pending'] else None
        result.generations = state['generations']
        result.rng.setstate(_tuples(state['rng_state']))
        result.bo_rng.setstate(_tuples(state['bo_rng_state']))
        result.mutation_multiplier = state['mutation_multiplier']
        result.stagnation, result.previous_hv = state['stagnation'], state['previous_hv']
        return result
