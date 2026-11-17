from pathlib import Path
import json,csv,sqlite3,math,statistics,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
RET=ROOT/'03_results/native_restricted'
C=RET/'campaign_artifacts/campaign-v2'
OUT=ROOT/'01_documentation';FIG=OUT/'figures';TAB=OUT/'data_tables'
FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(exist_ok=True)
M=['syftr_motpe','nsga2','hybrid_motpe','adaptive_nsga2'];LABEL=['MOTPE','NSGA II','Hybrid','Adaptive'];COL=['#3465a4','#d48b28','#248769','#8b5da8'];names=dict(zip(M,LABEL))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':150,'savefig.dpi':200})
read=lambda p:json.loads(Path(p).read_text())
F=read(C/'comparison/findings.json');A=read(RET/'AUDIT_at_25.json')
alltr=[];qa=[];curves={};generations={}
for m in M:
 run=C/'runs'/f'{m}-seed11'
 db=sqlite3.connect(f'file:{run}/study.sqlite3?mode=ro',uri=True)
 for tid,num,state in db.execute('SELECT trial_id,number,state FROM trials ORDER BY number'):
  attrs={k:json.loads(v) for k,v in db.execute('SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?',(tid,))}
  vals=[v for _,v in db.execute('SELECT objective,value FROM trial_values WHERE trial_id=? ORDER BY objective',(tid,))]
  r={'method':m,'trial':num,'state':state,'origin':attrs.get('proposal_origin','baseline'),'params':attrs.get('concrete_params',{}),'values':vals,'metrics':attrs.get('metrics',{}),'wall_seconds':attrs.get('wall_seconds',0),'failure_reason':attrs.get('failure_reason'),'candidate_id':attrs.get('candidate_id')}
  alltr.append(r)
  if 'result_path' in attrs:
   for line in (run/attrs['result_path']/'questions.jsonl').read_text().splitlines():qa.append({'method':m,'trial':num,**json.loads(line)})
 db.close()
 with (run/'analysis/convergence.csv').open() as f:curves[m]=list(csv.DictReader(f))
 if (run/'optimizer_generation_log.json').exists():generations[m]=read(run/'optimizer_generation_log.json')
complete=[r for r in alltr if r['state']=='COMPLETE'];assert len(complete)==98 and len(qa)==980
# Export flat measured trial table and exact active parameter table for review.
flat=[{'method':r['method'],'trial':r['trial'],'state':r['state'],'origin':r['origin'],'f1':r['values'][0] if r['values'] else None,'tokens_per_qa':r['values'][1] if r['values'] else None,'wall_seconds':r['wall_seconds'],'failure_reason':r['failure_reason'],'params_json':json.dumps(r['params'],sort_keys=True)} for r in alltr]
def csvout(name,rows):
 with (TAB/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvout('all_100_attempts.csv',flat)
(TAB/'measured_records.json').write_text(json.dumps({'trials':alltr,'questions':qa,'generations':generations},indent=2))
def save(name,fig):
 fig.savefig(FIG/(name+'.png'),bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8.2,4.5))
for m,l,col in zip(M,LABEL,COL):
 ps=[r['values'] for r in complete if r['method']==m];ax.scatter([v[1] for v in ps],[v[0] for v in ps],s=20,alpha=.25,c=col)
 fr=sorted(set((x['generator_tokens_per_qa'],x['f1']) for x in F['fronts'] if x['method']==m))
 ax.plot([x for x,y in fr],[y for x,y in fr],'-o',label=l,c=col,ms=4)
ax.set(xscale='log',xlabel='Mean generator prompt plus completion tokens per QA  log scale',ylabel='Mean raw answer F1',title='Observed Pareto fronts at 25 attempts per method');ax.legend();ax.grid(alpha=.18);save('01_pareto',fig)
for name,x,label in [('02_convergence','attempt','Attempted configurations including duplicates'),('03_token_convergence','known_generator_tokens','Cumulative generator search tokens')]:
 fig,ax=plt.subplots(figsize=(8.2,4.1))
 for m,l,col in zip(M,LABEL,COL):ax.step([float(r[x]) for r in curves[m]],[float(r['hypervolume']) for r in curves[m]],where='post',label=l,c=col)
 if x=='attempt':ax.axvspan(1,10,color='grey',alpha=.10);ax.axvline(10,c='grey',ls=':',lw=1)
 ax.set(xlabel=label,ylabel='Normalized hypervolume',title='Search progress on the fixed ten question set');ax.legend(ncol=2);ax.grid(alpha=.2);save(name,fig)
fig,axes=plt.subplots(1,2,figsize=(8.2,3.5));rs=A['methods']
for ax,key,title in zip(axes,['hypervolume','best_f1'],['Final hypervolume','Best observed F1']):
 vals=[r[key] for r in rs];ax.bar(LABEL,vals,color=COL,width=.65);ax.set_ylim(0,.53);ax.set_title(title);ax.tick_params(axis='x',labelrotation=15)
 for i,v in enumerate(vals):ax.text(i,v+.008,f'{v:.4f}',ha='center',fontsize=9)
fig.tight_layout();save('04_final_scores',fig)
fig,axes=plt.subplots(2,2,figsize=(8.2,5.4),sharex=True,sharey=True)
for ax,m,l,col in zip(axes.flat,M,LABEL,COL):
 ax.hist([r['values'][0] for r in complete if r['method']==m],bins=np.linspace(0,.5,11),color=col,edgecolor='white');ax.set_title(l);ax.set_xlabel('Candidate mean F1');ax.set_ylabel('Completed candidates')
fig.tight_layout();save('05_f1_histograms',fig)
fig,axes=plt.subplots(2,2,figsize=(8.2,5.4),sharex=True)
for ax,m,l,col in zip(axes.flat,M,LABEL,COL):
 vals=[r['values'][1] for r in complete if r['method']==m];ax.hist(vals,bins=np.geomspace(300,max(r['values'][1] for r in complete)*1.01,10),color=col,edgecolor='white');ax.set_xscale('log');ax.set_title(l);ax.set_xlabel('Tokens per QA  log scale');ax.set_ylabel('Completed candidates')
fig.tight_layout();save('06_token_histograms',fig)
sel=read(C/'comparison/confirmation_selection.json');selected={s['method']:s['trial'] for s in sel};ids=read(C/'runs/syftr_motpe-seed11/questions_manifest.json')['selected_ids'];matrix=np.array([[next(q['f1'] for q in qa if q['method']==m and q['trial']==selected[m] and q['qa_id']==qid) for qid in ids] for m in M])
fig,ax=plt.subplots(figsize=(8.2,3.4));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');ax.set_yticks(range(4),[f'{l} trial {selected[m]}' for m,l in zip(M,LABEL)]);ax.set_xticks(range(10),[f'Q{i+1}' for i in range(10)]);ax.set_title('Search question F1 for the frozen finalists');
for i in range(4):
 for j in range(10):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.6 else 'black')
fig.colorbar(im,ax=ax,label='Raw answer F1');fig.tight_layout();save('07_question_heatmap',fig)
fig,axes=plt.subplots(1,2,figsize=(8.2,3.7));axes[0].bar(LABEL,[r['known_generator_tokens']/1000 for r in rs],color=COL);axes[0].set(ylabel='Thousands of generator tokens',title='Total search consumption');axes[1].bar(LABEL,[r['known_judge_tokens']/1000 for r in rs],color=COL);axes[1].set(ylabel='Thousands of judge tokens',title='Diagnostic judge consumption')
for ax in axes:ax.tick_params(axis='x',labelrotation=15)
fig.tight_layout();save('08_consumption',fig)
mods=['few_shot_enabled','hyde_enabled','reranker_enabled','additional_context_enabled','rag_query_decomposition_enabled'];mat=np.array([[sum(bool(r['params'].get(k,False)) for r in alltr if r['method']==m) for k in mods] for m in M])
fig,ax=plt.subplots(figsize=(8.2,3.8));im=ax.imshow(mat,cmap='Blues',aspect='auto',vmin=0,vmax=25);ax.set_yticks(range(4),LABEL);ax.set_xticks(range(5),['Few shot','HyDE','Reranker','Neighbors','Decomposition'],rotation=15);ax.set_title('Module activation among all 25 attempted configurations')
for i in range(4):
 for j in range(5):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>14 else 'black')
fig.colorbar(im,ax=ax,label='Attempts enabled');fig.tight_layout();save('09_modules',fig)
fig,axes=plt.subplots(1,2,figsize=(8.2,3.7))
for m,l,col in zip(M[1:],LABEL[1:],COL[1:]):
 g=generations[m];axes[0].plot([x['generation'] for x in g],[x['conditional_diversity'] for x in g],'-o',label=l,c=col);axes[1].plot([x['generation'] for x in g],[x['mutation_multiplier_used'] for x in g],'-o',label=l,c=col,alpha=.8)
axes[0].set(xlabel='Generation',ylabel='Conditional population diversity',ylim=(0,1));axes[1].set(xlabel='Generation',ylabel='Mutation multiplier used',ylim=(.8,1.2));axes[0].legend(fontsize=8);axes[1].set_title('All multipliers remained at one');fig.tight_layout();save('10_operator_diagnostics',fig)
# Same concrete configuration across arms: range of recorded mean F1.
groups={}
for r in complete:groups.setdefault(r['candidate_id'],[]).append(r)
variation=[]
for cid,rr in groups.items():
 if len({x['method'] for x in rr})>1:
  variation.append({'candidate_id':cid,'evaluations':len(rr),'methods':','.join(sorted({x['method'] for x in rr})),'f1_min':min(x['values'][0] for x in rr),'f1_max':max(x['values'][0] for x in rr),'f1_range':max(x['values'][0] for x in rr)-min(x['values'][0] for x in rr)})
variation.sort(key=lambda r:-r['f1_range']);csvout('repeated_configuration_variation.csv',variation)
fig,ax=plt.subplots(figsize=(8.2,3.9));top=variation[:10];ax.barh([x['candidate_id'][:8] for x in top][::-1],[x['f1_range'] for x in top][::-1],color='#946c51');ax.set(xlabel='Observed max minus min mean F1 for identical active parameters',ylabel='Configuration hash prefix',title='Repeated configuration variability across method runs');fig.tight_layout();save('11_repeat_variation',fig)
# Workflow breakdown and candidate timings are descriptive, not speedup evidence.
fig,axes=plt.subplots(1,2,figsize=(8.2,3.7))
for mode,col in [('rag','#3465a4'),('sub_question_rag','#d48b28')]:
 counts=[sum(r['params'].get('rag_mode')==mode for r in alltr if r['method']==m) for m in M];bottom=[sum(r['params'].get('rag_mode')=='rag' for r in alltr if r['method']==m) for m in M] if mode!='rag' else None
 axes[0].bar(LABEL,counts,bottom=bottom,label=mode,color=col)
axes[0].legend(fontsize=8);axes[0].set_ylabel('Attempted configurations');axes[1].boxplot([[r['wall_seconds'] for r in complete if r['method']==m] for m in M],tick_labels=LABEL);axes[1].set_ylabel('Candidate wall seconds');
for ax in axes:ax.tick_params(axis='x',labelrotation=15)
fig.tight_layout();save('12_workflows_timing',fig)
# Architecture infographic with code-owned boundaries.
fig,ax=plt.subplots(figsize=(8.2,4.4));ax.axis('off')
boxes=[(.02,.65,.28,.24,'Pinned HotPotQA\n4972 grounding documents\n10 fixed search questions'),(.36,.65,.28,.24,'One common evaluator\nNative Syftr flow builder\nQwen on GPU0  BGE on GPU1'),(.70,.65,.28,.24,'Measured outcomes\nRaw answer F1 and EM\nJournaled generator tokens'),(.02,.14,.28,.24,'Four search policies\nMOTPE  NSGA II\nHybrid  Adaptive'),(.36,.14,.28,.24,'25 attempts per method\nSame first 10 configurations\nIndependent re evaluation'),(.70,.14,.28,.24,'Verified evidence\n98 candidates  980 QA\nPareto fronts\nand convergence')]
for x,y,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor='#edf2f7',edgecolor='#8096a8'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
for a,b in [((.30,.77),(.36,.77)),((.64,.77),(.70,.77)),((.16,.38),(.16,.65)),((.30,.26),(.36,.26)),((.64,.26),(.70,.26)),((.5,.38),(.5,.65))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#506477','lw':1.5})
save('13_architecture',fig)
# compact figures from audited compact measurements copied separately in packaging
summary={'methods':rs,'fronts':F['fronts'],'frozen_selections':sel,'repeat_variation':variation,'generation_logs':generations,'search_ids':ids,'total_generator_tokens':sum(r['known_generator_tokens'] for r in rs),'total_judge_tokens':sum(r['known_judge_tokens'] for r in rs),'best_by_origin':{m:{o:max((r['values'][0] for r in complete if r['method']==m and r['origin']==o),default=None) for o in {r['origin'] for r in alltr if r['method']==m}} for m in M}}
(TAB/'report_data.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'figures':len(list(FIG.glob('*.png'))),'total_generator_tokens':summary['total_generator_tokens'],'total_judge_tokens':summary['total_judge_tokens'],'repeat_max':variation[0],'best_by_origin':summary['best_by_origin']},indent=2))

COMPACT=ROOT/'03_results/compact/artifacts/compact_kcb_results'
land=read(COMPACT/'search_qa_5/landscape.json');replay=read(COMPACT/'search_qa_5/optimizer_replays/replay_results.json')
fig,ax=plt.subplots(figsize=(8.2,3.8))
for k,col in zip([2,4,6],COL[:3]):
 pts=[x['values'] for x in land.values() if x['params']['k']==k]
 ax.scatter([x[1] for x in pts],[x[0] for x in pts],label=f'k = {k}',color=col,s=38,alpha=.65)
front=sorted(replay['measured_grid_front'],key=lambda x:x[1]);ax.plot([x[1] for x in front],[x[0] for x in front],'-o',c='black',label='Measured grid front')
ax.set(xlabel='Generator tokens per QA',ylabel='Mean search F1',title='Compact measured landscape on five search questions');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();save('14_compact_front',fig)
fig,axes=plt.subplots(1,2,figsize=(8.2,3.8));rr=replay['methods']
axes[0].bar(LABEL,[x['hypervolume_mean'] for x in rr],color=COL);axes[0].set(ylim=(0,.65),ylabel='Mean final HV',title='All four methods tie')
axes[1].bar(LABEL,[x['unique_configurations_mean'] for x in rr],color=COL);axes[1].set(ylim=(0,27),ylabel='Mean distinct settings',title='Different exploration counts')
for ax in axes:ax.tick_params(axis='x',labelrotation=15)
fig.tight_layout();save('15_compact_replay',fig)
