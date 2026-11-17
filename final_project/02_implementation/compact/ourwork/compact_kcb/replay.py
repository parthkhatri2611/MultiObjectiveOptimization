"""Optimizers see only requested cells of an actual measured finite landscape."""
from pathlib import Path
import itertools,json,random,statistics,time
import optuna
from optuna.distributions import CategoricalDistribution
from ourwork.optimizers.space import ConditionalSpace
from ourwork.optimizers.nsga2 import EvolutionSearch
from ourwork.baseline.records import atomic_json,fingerprint
from ourwork.baseline.metrics import hypervolume,pareto_indices,igd

METHODS=['syftr_style_motpe','nsga2','hybrid_motpe','adaptive_nsga2']


class GridSpace:
    def __init__(self,grid):self.grid=grid
    def sample(self,trial):return {k:trial.suggest_categorical(k,v) for k,v in self.grid.items()}
    def build_distributions(self,params=None):return {k:CategoricalDistribution(v) for k,v in self.grid.items()}


class GridCodec(ConditionalSpace):
    @staticmethod
    def block(key):return key


def grid_points(protocol):
    keys=list(protocol['grid'])
    return [dict(zip(keys,v)) for v in itertools.product(*(protocol['grid'][k] for k in keys))]


def run_replays(table,protocol,output):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    configs=grid_points(protocol);codec=GridCodec(GridSpace(protocol['grid']))
    valid=[tuple(row['values']) for row in table.values() if row.get('values') is not None]
    reference=sorted(set(valid[i] for i in pareto_indices(valid)))
    ref=protocol['hypervolume_reference'];gold_hv=hypervolume(valid,ref)
    all_results=[]
    for seed in protocol['optimizer_seeds']:
        initial=random.Random(seed).sample(configs,protocol['startup'])
        for method in METHODS:
            start=time.monotonic();engine=None;study=None
            if method=='syftr_style_motpe':
                sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=protocol['startup'],multivariate=True,constant_liar=True)
                study=optuna.create_study(sampler=sampler,directions=['maximize','minimize'])
                for p in initial:study.enqueue_trial(p)
            else:
                engine=EvolutionSearch(codec,initial,seed,method,protocol['attempts_per_optimizer'],protocol['population'],ref)
            trace=[];observed=[];seen=set();equivalent_tokens=0
            for attempt in range(protocol['attempts_per_optimizer']):
                if study is not None:
                    trial=study.ask();params=codec.space.sample(trial);origin='shared_initialization' if attempt<protocol['startup'] else 'motpe'
                else:
                    proposed=engine.ask();params=proposed.params;origin=proposed.origin
                cid=fingerprint(params);measurement=table[cid];values=measurement.get('values')
                repeated=cid in seen;seen.add(cid)
                # One charged black-box query; the optimizer is never given the rest of the table.
                if study is not None:
                    study.tell(trial,values) if values is not None else study.tell(trial,state=optuna.trial.TrialState.FAIL)
                else:engine.tell(values,None if values is not None else 'ineligible_measured_configuration')
                if values is not None:observed.append(tuple(values))
                equivalent_tokens+=measurement.get('known_generator_tokens',0)
                trace.append({'attempt':attempt+1,'configuration_id':cid,'params':params,'values':values,'proposal_origin':origin,
                    'repeat_query':repeated,'unique_configurations':len(seen),'hypervolume':hypervolume(observed,ref),
                    'equivalent_generator_tokens':equivalent_tokens,'physical_generator_calls_during_replay':0})
            front=sorted(set(observed[i] for i in pareto_indices(observed)))
            result={'method':method,'seed':seed,'trace':trace,'front':front,'hypervolume':hypervolume(observed,ref),
                'hv_gap_to_measured_grid':gold_hv-hypervolume(observed,ref),'igd_to_measured_grid_front':igd(observed,reference,ref),
                'front_objective_recovery_fraction':len(set(front)&set(reference))/len(reference) if reference else None,
                'unique_configurations':len(seen),'repeat_queries':sum(t['repeat_query'] for t in trace),
                'ineligible_queries':sum(t['values'] is None for t in trace),'equivalent_generator_tokens':equivalent_tokens,
                'cpu_optimizer_seconds':time.monotonic()-start,'shared_initialization_sha256':fingerprint(initial),
                'generation_log':engine.generations if engine else [],'optimizer_state':engine.state_dict() if engine else None}
            atomic_json(output/f'{method}_seed{seed}.json',result);all_results.append(result)
    summary=[]
    for method in METHODS:
        rows=[r for r in all_results if r['method']==method]
        summary.append({'method':method,'seeds':len(rows),**{
            key+'_mean':statistics.mean(r[key] for r in rows) for key in
            ['hypervolume','hv_gap_to_measured_grid','unique_configurations','repeat_queries','equivalent_generator_tokens']},
            'hypervolume_std':statistics.stdev(r['hypervolume'] for r in rows) if len(rows)>1 else None,
            'igd_to_measured_grid_front_mean':statistics.mean(r['igd_to_measured_grid_front'] for r in rows)
                if all(r['igd_to_measured_grid_front'] is not None for r in rows) else None})
    record={'methods':summary,'measured_grid_front':reference,'measured_grid_hv':gold_hv,'valid_configurations':len(valid),
        'total_configurations':len(configs),'global_grid_front_identified':len(valid)==len(configs),
        'scope':'Optimizer replay on fixed measured fitness, not independent end-to-end GPU reruns.',
        'landscape_sha256':fingerprint(table),'results':all_results}
    atomic_json(output/'replay_results.json',record)
    return record
