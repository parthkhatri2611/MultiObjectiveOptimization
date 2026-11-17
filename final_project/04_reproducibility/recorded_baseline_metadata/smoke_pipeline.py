import os, signal, subprocess, time, urllib.request, shutil
from pathlib import Path
ROOT=Path('/kaggle/working/syftr_project')
OUTPUT=Path('/kaggle/working/baseline_results')
SCRATCH=Path('/kaggle/temp/rm-syftr')
PYTHON=str(SCRATCH/'venv/bin/python')
print('WAITING_FOR_PINNED_CUDA_BUILD',flush=True)
for i in range(360):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=3) as response:
            if response.status==200: break
    except OSError: time.sleep(10)
else: raise TimeoutError('Existing CUDA build did not become healthy')
print('INITIAL_SERVER_HEALTHY',flush=True)
os.killpg(516,signal.SIGTERM)
time.sleep(3)
with (OUTPUT/'server.log').open('a') as log:
    server=subprocess.Popen([PYTHON,'-m','ourwork.baseline.serve','--workspace',str(SCRATCH/'server'),'--metadata',str(OUTPUT/'server_metadata.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for i in range(240):
        if server.poll() is not None: raise RuntimeError('Persistent server failed')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=3) as response:
                if response.status==200: break
        except OSError: time.sleep(5)
    else: raise TimeoutError('Persistent server readiness failed')
    print('PERSISTENT_SERVER_HEALTHY',server.pid,flush=True)
    (OUTPUT/'server.pid').write_text(str(server.pid))
    subprocess.run([PYTHON,'-m','ourwork.baseline.preflight','--output',str(OUTPUT/'preflight.json')],cwd=ROOT,check=True)
    print('INFERENCE_PREFLIGHT_PASSED',flush=True)
    subprocess.run([PYTHON,'-m','ourwork.baseline.run','--protocol',str(ROOT/'ourwork/baseline/configs/smoke.json'),'--output',str(OUTPUT/'smoke-seed11'),'--workspace',str(SCRATCH/'evaluation'),'--seed','11','--max-hours','1','--server-metadata',str(OUTPUT/'server_metadata.json'),'--preflight',str(OUTPUT/'preflight.json')],cwd=ROOT,check=True)
    print('SMOKE_FINISHED',flush=True)
    shutil.make_archive('/kaggle/working/baseline_checkpoint','zip',OUTPUT)
