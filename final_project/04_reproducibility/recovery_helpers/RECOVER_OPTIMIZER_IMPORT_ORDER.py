"""Run this single cell in the existing Kaggle session. Does not reinstall or edit research code."""
from pathlib import Path
import subprocess, json, sys, time, hashlib
root=Path('/kaggle/working/syftr_project')
campaign=Path('/kaggle/working/baseline_results/campaign-v2')
python='/kaggle/temp/rm-syftr/venv/bin/python'
assert (campaign/'protocol.json').exists(), 'Restore the existing campaign first.'
assert json.loads((campaign/'status.json').read_text())['status'] in {'error','paused'}, 'A campaign is still active; do not launch twice.'
launcher=Path('/kaggle/working/recover_optimizer_import_order.py')
code=r'''
import subprocess, sys, json, hashlib, fcntl
from pathlib import Path
root=Path('/kaggle/working/syftr_project')
sys.path.insert(0,str(root))
from ourwork.campaign_v2.common import implementation_identity, read, atomic
campaign=Path('/kaggle/working/baseline_results/campaign-v2')
assert implementation_identity()==read(campaign/'implementation.json'), 'Research source changed.'
import sqlite3, time
lock=(campaign/'.controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
quarantined=[]
try:
    for method in ['nsga2','hybrid_motpe','adaptive_nsga2']:
        run=campaign/'runs'/f'{method}-seed11'
        if not (run/'protocol.json').exists() or (run/'sampler_checkpoint.json').exists():
            continue
        db=run/'study.sqlite3'
        if db.exists():
            with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as connection:
                tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                count=connection.execute('SELECT COUNT(*) FROM trials').fetchone()[0] if 'trials' in tables else 0
            if count:
                raise RuntimeError(f'{method}: existing trials without sampler state; refusing to reset.')
        if (run/'analysis/summary.json').exists() and read(run/'analysis/summary.json').get('attempts',0):
            raise RuntimeError(f'{method}: measured attempts exist; refusing to reset.')
        if any(run.rglob('result.json')):
            raise RuntimeError(f'{method}: candidate results exist; refusing to reset.')
        target=campaign/'initialization_failures'/f'{method}-seed11-{time.time_ns()}'
        target.parent.mkdir(parents=True,exist_ok=True)
        run.rename(target)
        quarantined.append(str(target))
        print('PRESERVED EMPTY INITIALIZATION:',target,flush=True)
finally:
    lock.close()
atomic(campaign/'initialization_recovery.json',{'preserved_folders':quarantined,'policy':'Only zero-trial folders with missing sampler state are moved; baseline never modified.'})
original=subprocess.Popen
preload="""
import sys, json
from pathlib import Path
sys.path.insert(0,'/kaggle/working/syftr_project')
a=sys.argv[1:]
def value(flag): return a[a.index(flag)+1]
from ourwork.baseline.spec import load_protocol
from ourwork.baseline.adapter import configure
cap=json.loads((Path(value('--capabilities-run'))/'protocol.json').read_text())
configure(load_protocol(value('--protocol')),value('--workspace'),cap['sources'])
# Define Syftr's subclass while TPESampler is still the real Optuna class.
import syftr.hierarchical_tpe
from ourwork.campaign_v2.run import main
main()
"""
def launch(command,*args,**kwargs):
    if isinstance(command,list) and '-m' in command and command[command.index('-m')+1]=='ourwork.campaign_v2.run':
        i=command.index('-m')
        command=command[:i]+['-c',preload]+command[i+2:]
    return original(command,*args,**kwargs)
subprocess.Popen=launch
atomic(campaign/'import_order_recovery.json',{'change':'Pre-import Syftr hierarchical_tpe before the temporary sampler-factory patch', 'research_source_unchanged':True,'implementation':implementation_identity(), 'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
from ourwork.campaign_v2.controller import control
control(hours=8)
'''
launcher.write_text(code)
log=Path('/kaggle/working/import_order_recovery.log')
with log.open('a') as stream:
    process=subprocess.Popen([python,'-u',str(launcher)],cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print('RECOVERY STARTED',process.pid,'Existing completed candidates are retained.')
print('Log:',log)
print('Run the progress cell below separately; do not rerun this launch cell.')
