"""Measure the finite grid progressively, replay optimizers, confirm frozen selections."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,statistics,subprocess,time,zipfile
from ourwork.baseline.records import atomic_json,fingerprint,read_jsonl
from ourwork.baseline.metrics import hypervolume
from .measure import dataset_data,DenseRetrieval,Generator,measure_config
from .replay import grid_points,run_replays
from .report import report,write_csv
from .audit import audit_table

OUT=Path('/kaggle/working/compact_kcb_results')


def export(out=OUT):
    target=Path('/kaggle/working/compact_kcb_review.zip');tmp=target.with_suffix('.zip.tmp')
    inventory={}
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.endswith(('.lock','.tmp')):
                name=str(Path('compact_kcb_results')/p.relative_to(out));data=p.read_bytes()
                z.writestr(name,data);inventory[name]=hashlib.sha256(data).hexdigest()
        z.writestr('CONTENT_MANIFEST.json',json.dumps(inventory,indent=2))
    tmp.replace(target)
    h=hashlib.sha256()
    with target.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    atomic_json(target.with_name('compact_kcb_receipt.json'),{'file':target.name,'bytes':target.stat().st_size,'sha256':h.hexdigest(),
        'saved_unix':time.time(),'scope':'Actual compact measurements and explicitly labeled optimizer replays; no model weights/index cache.'})


def run(hours=3):
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.setdefault('MPLCONFIGDIR','/kaggle/temp/compact_kcb/matplotlib')
    source=Path(__file__).resolve().parent
    protocol=json.loads((source/'protocol.json').read_text())
    root=source.parents[1]
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['compact_kcb','optimizers','baseline'] for p in (root/'ourwork'/directory).rglob('*') if p.suffix in {'.py','.json'}}
    identity={'protocol':protocol,'compact_source':files,'method':'fixed measured finite-table optimizer replay'}
    if (OUT/'protocol.json').exists() and json.loads((OUT/'protocol.json').read_text())!=identity:raise ValueError('Saved compact protocol/code differs')
    atomic_json(OUT/'protocol.json',identity)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(m in args for m in ['ourwork.campaign_v2.controller','ourwork.campaign_v2.run','ourwork.baseline.run','ourwork.baseline.worker']):
            raise RuntimeError('Another research experiment is active. Use a separate GPU session for compact study.')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    selected=[r.split(',')[1].strip() for r in gpu.splitlines() if r.split(',')[0].strip()=='1']
    if len(selected)!=1:raise ValueError('Use T4 x2: generator on physical GPU0, embedding on physical GPU1')
    os.environ['CUDA_VISIBLE_DEVICES']=selected[0];os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import torch
    if not torch.cuda.is_available():raise ValueError('GPU embedding device unavailable')
    cache=Path('/kaggle/temp/compact_kcb_cache');cache.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(protocol['tokenizer_repo'],revision=protocol['tokenizer_revision'],cache_dir=str(cache/'hf'))
    embedding=SentenceTransformer(protocol['embedding_model'],revision=protocol['embedding_revision'],device='cuda:0',cache_folder=str(cache/'hf'),trust_remote_code=False)
    embedding.max_seq_length=protocol['embedding_max_sequence_tokens']
    data,documents=dataset_data(protocol,cache/'hf')
    manifest={'selected_qa':data,'corpora':{p:{'documents':len(v),'sha256':fingerprint(v)} for p,v in documents.items()}}
    if (OUT/'data_manifest.json').exists() and json.loads((OUT/'data_manifest.json').read_text())!=manifest:raise ValueError('Prepared data changed')
    atomic_json(OUT/'data_manifest.json',manifest)
    atomic_json(OUT/'environment.json',{'gpu':gpu,'embedding_visible_uuid':selected[0],'torch':torch.__version__,
        'packages':subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True)})
    retrieval=DenseRetrieval(documents,data,tokenizer,embedding,protocol,cache/'indexes')
    generator=Generator(protocol,OUT/'endpoint')
    deadline=time.monotonic()+hours*3600;last=None;last_table=None
    try:
        for qa_count in protocol['search_qa_stages']:
            table={}
            for i,params in enumerate(grid_points(protocol)):
                if time.monotonic()>deadline:
                    atomic_json(OUT/'status.json',{'status':'paused','last_completed_search_qa_stage':last,'note':'No incomplete grid is presented as complete.'});return
                cid=fingerprint(params)
                result=measure_config(params,'test',data['test'][:qa_count],retrieval,generator,OUT/'measurements/test'/cid)
                table[cid]=result
                atomic_json(OUT/'status.json',{'status':'measuring','search_qa_stage':qa_count,'completed_configurations':i+1,'total_configurations':27})
                export()
            audit_table(table,tokenizer)
            stage=OUT/f'search_qa_{qa_count}'
            atomic_json(stage/'landscape.json',table)
            replays=run_replays(table,protocol,stage/'optimizer_replays')
            report(table,replays,protocol,stage,qa_count)
            last=qa_count;last_table=table
            atomic_json(OUT/'status.json',{'status':'search_checkpoint_complete','search_qa_stage':qa_count,'configurations':27,'optimizer_episodes':len(replays['results'])})
            export()
        selection=[]
        for result in replays['results']:
            eligible=[t for t in result['trace'] if t['values'] is not None]
            if not eligible:continue
            chosen=max(eligible,key=lambda t:(hypervolume([tuple(t['values'])],protocol['hypervolume_reference']),t['values'][0],-t['values'][1],-t['attempt']))
            selection.append({'method':result['method'],'seed':result['seed'],'configuration_id':chosen['configuration_id'],'params':chosen['params']})
        frozen=OUT/'holdout_selection.json'
        if frozen.exists() and json.loads(frozen.read_text())!=selection:raise ValueError('Frozen holdout selection differs')
        atomic_json(frozen,selection)
        held={}
        for s in selection:
            cid=s['configuration_id']
            if cid in held:continue
            if time.monotonic()>deadline:
                atomic_json(OUT/'status.json',{'status':'paused_during_holdout','search_qa_count':last,'confirmed_unique_configurations':len(held),'planned_unique_configurations':len({r['configuration_id'] for r in selection})});return
            held[cid]=measure_config(s['params'],'holdout',data['holdout'],retrieval,generator,OUT/'measurements/holdout'/cid)
            atomic_json(OUT/'holdout_measurements.json',held);export()
        holdout_rows=[dict(s,values=held[s['configuration_id']]['values'],exact_match=held[s['configuration_id']]['exact_match']) for s in selection]
        write_csv(OUT/'holdout_results.csv',holdout_rows)
        physical=read_jsonl(OUT/'endpoint/calls.jsonl')
        atomic_json(OUT/'physical_cost.json',{'actual_endpoint_calls':len(physical),'actual_known_tokens':sum(r['usage']['prompt_tokens']+r['usage']['completion_tokens'] for r in physical if r.get('usage')),
            'failed_calls':sum(bool(r['error']) for r in physical),'optimizer_replay_physical_calls':0,
            'scope':'Repeated logical queries reuse measured fitness; equivalent search tokens are not additional physical token expenditure.'})
        atomic_json(OUT/'status.json',{'status':'complete','search_qa_count':last,'grid_cells':27,'replay_seeds':len(protocol['optimizer_seeds']),
            'holdout_selected_unique_configurations':len(held),'physical_endpoint_calls':len(physical)})
    finally:export();lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=3);args=parser.parse_args();run(args.hours)
