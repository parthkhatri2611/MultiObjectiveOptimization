from pathlib import Path
import csv,json,statistics
from ourwork.baseline.records import atomic_json
from .replay import METHODS


def write_csv(path,rows):
    keys=sorted({k for r in rows for k in r})
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def report(table,replays,protocol,out,qa_count):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'configurations.csv',[{'configuration_id':cid,**r['params'],'eligible':r['values'] is not None,
        'f1':r['values'][0] if r['values'] else None,'generator_tokens_per_qa':r['values'][1] if r['values'] else None,
        'exact_match':r.get('exact_match'),'questions':qa_count, 'known_generator_tokens':r.get('known_generator_tokens')}
        for cid,r in table.items()])
    write_csv(out/'optimizer_summary.csv',replays['methods'])
    write_csv(out/'attempts.csv',[{'method':r['method'],'seed':r['seed'],**t} for r in replays['results'] for t in r['trace']])
    means=[]
    for key,choices in protocol['grid'].items():
        for value in choices:
            members=[r for r in table.values() if r['params'][key]==value and r['values'] is not None]
            means.append({'parameter':key,'value':value,'eligible_cells':len(members),
                          'mean_f1':statistics.mean(r['values'][0] for r in members) if members else None,
                          'mean_tokens':statistics.mean(r['values'][1] for r in members) if members else None})
    write_csv(out/'factorial_parameter_means.csv',means)
    lines=['# Controlled k / c / Bc RAG experiment','',f'Search questions per configuration: {qa_count}.',
        'Optimizer results are replays on a fixed table of actual measured answers and token usage. No independent GPU-speedup claim.',
        'At one question this is a smoke result only. Three questions are preliminary; five are the planned small-sample study.','',
        '| Method | Mean HV | HV std | Mean unique queries |','|---|---:|---:|---:|']
    for r in replays['methods']:lines.append(f'| {r["method"]} | {r["hypervolume_mean"]:.6f} | {r["hypervolume_std"]} | {r["unique_configurations_mean"]:.2f} |')
    lines+=['',f'Measured eligible grid cells: {replays["valid_configurations"]}/27.',
        'The finite-grid frontier is exact only if all 27 cells are eligible; otherwise it is the frontier of valid measured cells.',
        'Optimizer-seed variability is conditional on this small fixed question set and measured table.',
        'This is a Syftr-derived controlled design with an added context budget, not the native/full Syftr search space.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    gold=replays['measured_grid_front']
    ax.plot([c for q,c in sorted(gold,key=lambda v:v[1])],[q for q,c in sorted(gold,key=lambda v:v[1])],'k--',label='Measured grid frontier')
    for i,method in enumerate(METHODS):
        r=next(r for r in replays['results'] if r['method']==method and r['seed']==protocol['optimizer_seeds'][0])
        front=sorted(r['front'],key=lambda v:v[1]);color=plt.get_cmap('tab10')(i)
        ax.plot([c for q,c in front],[q for q,c in front],marker='o',color=color,label=method)
    ax.set(xlabel='Mean logical generator input + output tokens / QA',ylabel='Raw-answer F1',
           title=f'Controlled 27-cell RAG grid, {qa_count} search QA; representative seed {protocol["optimizer_seeds"][0]}')
    ax.set_ylim(-.02,1.02);ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'pareto.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for i,method in enumerate(METHODS):
        rows=[r for r in replays['results'] if r['method']==method]
        x=list(range(1,protocol['attempts_per_optimizer']+1))
        mean=[statistics.mean(r['trace'][j]['hypervolume'] for r in rows) for j in range(len(x))]
        sd=[statistics.stdev(r['trace'][j]['hypervolume'] for r in rows) if len(rows)>1 else 0 for j in range(len(x))]
        ax.plot(x,mean,label=method,color=plt.get_cmap('tab10')(i))
        ax.fill_between(x,[max(0,a-b) for a,b in zip(mean,sd)],[min(1,a+b) for a,b in zip(mean,sd)],alpha=.12)
    ax.axhline(replays['measured_grid_hv'],color='black',linestyle='--',label='Measured grid HV')
    ax.set(xlabel='Charged black-box queries (repeats included)',ylabel='Normalized hypervolume',title='Optimizer replay: mean ± one std across seeds')
    ax.set_ylim(-.02,1.02);ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'convergence.{ext}',dpi=180)
    plt.close(fig)
