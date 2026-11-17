"""Flat-asset compact setup. No native Syftr source restore or legacy setup tests."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,urllib.request

HERE=Path(__file__).resolve().parent
ROOT=Path('/kaggle/working/compact_kcb_project')
OUT=Path('/kaggle/working/compact_kcb_results')
SCRATCH=Path('/kaggle/temp/compact_kcb')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def progress(stage,**extra):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'setup_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'status':stage,'time':time.time(),**extra},indent=2)+'\n');t.replace(p)
    print(stage,extra,flush=True)


def active_other_experiment():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(m in args for m in ['ourwork.campaign_v2.controller','ourwork.campaign_v2.run','ourwork.baseline.run','ourwork.baseline.worker']):return int(p.name)
    return None


def main():
    progress('verifying_flat_assets')
    manifest=json.loads((HERE/'MANIFEST.json').read_text())
    for name,expected in manifest.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(HERE/name)==expected,name
    busy=active_other_experiment()
    if busy:raise RuntimeError(f'Another experiment is active (PID {busy}). Use a separate Kaggle GPU session; do not interfere with it.')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    rows=[r.split(',') for r in gpu.splitlines()]
    if len(rows)!=2 or any('T4' not in r[2] for r in rows):raise ValueError('Select GPU T4 x2')
    for name in manifest:
        if name.startswith('project/'):
            rel=Path(name).relative_to('project');target=ROOT/rel;target.parent.mkdir(parents=True,exist_ok=True)
            data=(HERE/name).read_bytes()
            if target.exists() and target.read_bytes()!=data:raise ValueError('Existing compact source differs: '+str(target))
            target.write_bytes(data)
    protocol=json.loads((ROOT/'ourwork/compact_kcb/protocol.json').read_text())
    versions={line.split('==')[0]:line.split('==')[1] for line in (HERE/'requirements.txt').read_text().splitlines() if line.strip()}
    python=Path('/kaggle/temp/rm-syftr/venv/bin/python')
    check='import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in '+repr(list(versions))+'}))'
    reuse=False
    if python.exists():
        candidate=subprocess.run([str(python),'-c',check],capture_output=True,text=True)
        if candidate.returncode==0:reuse=json.loads(candidate.stdout)==versions
    if not reuse:
        progress('installing_minimal_locked_environment')
        SCRATCH.mkdir(parents=True,exist_ok=True)
        packages=SCRATCH/'bootstrap_packages'
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(packages),'uv==0.9.6'],check=True)
        env=dict(os.environ,PYTHONPATH=str(packages),UV_CACHE_DIR=str(SCRATCH/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(SCRATCH/'python'))
        python=SCRATCH/'venv/bin/python'
        if not python.exists():subprocess.run([sys.executable,'-m','uv','venv','--seed','--python','3.12',str(SCRATCH/'venv')],env=env,check=True)
        subprocess.run([sys.executable,'-m','uv','pip','install','--python',str(python),'-r',str(HERE/'requirements.txt'),'-c',str(HERE/'constraints.txt')],env=env,check=True)
    for name in ['MANIFEST.json','requirements.txt','constraints.txt','bootstrap.py']:
        shutil.copy2(HERE/name,OUT/('package_'+name))
    progress('python_ready',executable=str(python),reused_existing_environment=reuse)
    subprocess.run([str(python),'-m','unittest','ourwork.compact_kcb.tests','-q'],cwd=ROOT,check=True)
    runtime=json.loads((HERE/'assets/runtime_manifest.json').read_text())
    metadata=json.loads((HERE/'assets/server_metadata.json').read_text())
    bin_dir=SCRATCH/'runtime/bin';blob_dir=SCRATCH/'runtime/blobs';bin_dir.mkdir(parents=True,exist_ok=True);blob_dir.mkdir(exist_ok=True)
    for name,digest in runtime['files_sha256'].items():
        local=blob_dir/(digest+'.bin')
        if not local.exists():shutil.copy2(HERE/'assets/runtime_blobs'/(digest+'.bin'),local)
        assert sha(local)==digest
        alias=bin_dir/name
        if not alias.exists():os.link(local,alias)
        assert sha(alias)==digest
    binary=bin_dir/'llama-server';binary.chmod(binary.stat().st_mode|0o111)
    ready=False
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=3) as response:ready=response.status==200
    except OSError:pass
    if ready:
        # Reuse only a known server with exactly the same pinned model files and runtime.
        candidates=[OUT/'server_runtime.json',Path('/kaggle/working/baseline_results/server_metadata.json')]
        found=False
        for p in candidates:
            if not p.exists():continue
            m=json.loads(p.read_text())
            if m.get('model_revision')==metadata['model_revision'] and m.get('files_sha256')==metadata['files_sha256'] and m.get('binary_sha256')==metadata['binary_sha256']:
                metadata=dict(m,reused_existing_server=True);found=True;break
        if not found:raise ValueError('Port 8765 belongs to an unrecognized model server')
        progress('reusing_pinned_model_server')
    else:
        progress('downloading_pinned_model_weights')
        download='''import json,hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download
m=json.loads(Path(__import__('sys').argv[1]).read_text());paths=[]
for name,expected in m['files_sha256'].items():
 p=Path(hf_hub_download(m['model_repo'],filename=name,revision=m['model_revision'],cache_dir='/kaggle/temp/compact_kcb/weights'))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==expected,name
 paths.append(str(p));print('WEIGHT_VERIFIED',name,flush=True)
Path('/kaggle/working/compact_kcb_results/model_paths.json').write_text(json.dumps(paths))
'''
        subprocess.run([str(python),'-u','-c',download,str(HERE/'assets/server_metadata.json')],check=True)
        model=json.loads((OUT/'model_paths.json').read_text())[0]
        command=[str(binary),'-m',model,'--alias',protocol['model_alias'],'--host','127.0.0.1','--port','8765','-c','16384',
                 '-ngl','99','--parallel','1','--jinja','--seed','20261114','--threads','2','--no-context-shift']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=next(r[1].strip() for r in rows if r[0].strip()=='0'))
        env['LD_LIBRARY_PATH']=':'.join([str(bin_dir),'/usr/local/nvidia/lib64','/usr/lib/x86_64-linux-gnu','/usr/lib64-nvidia','/usr/local/cuda/compat',env.get('LD_LIBRARY_PATH','')])
        with (OUT/'server.log').open('a') as log:server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        progress('starting_gpu0_server',pid=server.pid)
        for _ in range(180):
            if server.poll() is not None:raise RuntimeError('Saved CUDA server failed; inspect compact_kcb_results/server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=3) as response:ready=response.status==200
                if ready:break
            except OSError:pass
            time.sleep(2)
        if not ready:raise TimeoutError('Model server startup exceeded six minutes')
        metadata=dict(metadata,command=command,pid=server.pid)
    (OUT/'server_runtime.json').write_text(json.dumps(metadata,indent=2)+'\n')
    progress('preflight')
    body={'model':protocol['model_alias'],'temperature':0,'max_tokens':32,'messages':[{'role':'user','content':'Reply READY.'}]}
    request=urllib.request.Request('http://127.0.0.1:8765/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=120) as r:result=json.load(r)
    if not result['choices'][0]['message'].get('content') or not all(type(result.get('usage',{}).get(k)) is int for k in ['prompt_tokens','completion_tokens']):raise ValueError('Preflight lacks answer or reliable token usage')
    (OUT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    progress('running_compact_measurements')
    with (OUT/'run.log').open('a') as log:
        process=subprocess.Popen([str(python),'-u','-m','ourwork.compact_kcb.main','--hours','3'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while True:
            try:code=process.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                p=OUT/'status.json';print(p.read_text() if p.exists() else 'Preparing embeddings/data',flush=True)
    if code:raise RuntimeError('Compact measurement/replay failed; inspect compact_kcb_results/run.log')
    progress('finished',results='/kaggle/working/compact_kcb_review.zip')
    print((OUT/'status.json').read_text(),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        progress('error',error=f'{type(e).__name__}: {e}');raise
