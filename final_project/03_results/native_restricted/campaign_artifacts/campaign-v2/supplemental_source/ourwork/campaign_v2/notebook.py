"""Fast notebook controls; inference runs in managed background processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from .common import CAMPAIGN, OUTPUT, ROOT, PYTHON, METHODS, atomic, read


def status():
    setup = Path('/kaggle/working/v2_setup_status.json')
    if setup.exists():
        record = read(setup)
        print('SETUP:', json.dumps(record, indent=2))
        if record['status'] != 'ready':
            log = Path('/kaggle/working/v2_setup.log')
            if log.exists():
                print(log.read_text(errors='replace')[-2500:])
            return record
    path = CAMPAIGN / 'status.json'
    record = read(path) if path.exists() else {'status': 'ready_to_start'}
    record.pop('protocol', None)  # Keep progress output small enough for notebook limits.
    if record.get('heartbeat_unix'):
        record['heartbeat_age_seconds'] = round(time.time() - record['heartbeat_unix'], 1)
    print(json.dumps(record, indent=2))
    if record.get('active_run'):
        candidates = sorted((Path(record['active_run']) / 'candidates').glob('*'))
        if candidates:
            p = candidates[-1] / 'progress.json'
            if p.exists():
                print('ACTIVE CANDIDATE:', candidates[-1].name, 'progress_age_seconds:',
                      round(time.time()-p.stat().st_mtime,1), read(p))
            p = candidates[-1] / 'worker.log'
            if p.exists():
                print('WORKER:', p.read_text(errors='replace')[-1200:])
    confirmation = CAMPAIGN / 'confirmation/status.json'
    if confirmation.exists():
        heldout = read(confirmation)
        print('CONFIRMATION:', heldout)
        if heldout.get('active_method'):
            progress = CAMPAIGN / 'confirmation' / heldout['active_method'] / 'progress.json'
            if progress.exists():
                print('CONFIRMATION QA:', read(progress))
    export = Path('/kaggle/working/campaign_v2_exports/campaign_v2_receipt.json')
    if export.exists():
        print('LAST DOWNLOADABLE SNAPSHOT:', read(export))
    return record


def active_experiments():
    found = []
    modules = {'ourwork.campaign_v2.controller', 'ourwork.campaign_v2.run', 'ourwork.campaign_v2.confirm',
               'ourwork.baseline.run', 'ourwork.baseline.worker', 'ourwork.optimizers.run'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state != 'Z' and any(m in args for m in modules):
                found.append(int(proc.name))
        except OSError:
            pass
    return found


def launch(hours=8):
    if hours <= 0 or not read('/kaggle/working/v2_setup_status.json')['status'] == 'ready':
        raise ValueError('Wait for setup ready and choose positive hours')
    if active_experiments():
        raise RuntimeError('An experiment is already running; inspect status()')
    with urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=5) as response:
        assert response.status == 200
    stop = CAMPAIGN / 'STOP_REQUESTED'
    if stop.exists():
        stop.unlink()  # Resume explicitly requested by launch().
    log_path = CAMPAIGN / 'controller.log'
    with log_path.open('a') as log:
        process = subprocess.Popen([str(PYTHON), '-u', '-m', 'ourwork.campaign_v2.controller',
                                    '--hours', str(hours)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print('STARTED / RESUMED. PID:', process.pid)
    print('This cell is finished; the experiment continues in the session. Run status() anytime.')
    print('Use stop() to pause after the active candidate; download snapshots throughout the run.')
    return process.pid


def stop():
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / 'STOP_REQUESTED').write_text('User requested pause at next candidate boundary.\n')
    print('STOP REQUESTED. The active candidate may take time to finish; no later candidate will start.')
    print('Wait for status() to say paused, then download. This request does not stop the model server.')


def downloads():
    from IPython.display import FileLink, display
    for name in ['campaign_v2_review.zip', 'campaign_v2_receipt.json']:
        path = Path('/kaggle/working/campaign_v2_exports') / name
        if path.exists():
            print(path, 'bytes:', path.stat().st_size)
            display(FileLink(str(path)))
        else:
            print('Not created yet:', path)
    print('Refresh Output > campaign_v2_exports and download both. Remote snapshots are not local backups.')


def confirm(hours=1):
    if hours <= 0 or active_experiments():
        raise ValueError('Use positive hours and wait until all experiments have stopped')
    if not (CAMPAIGN / 'comparison/confirmation_selection.json').exists():
        raise ValueError('Finish and audit all four search budgets first')
    flag = CAMPAIGN / 'STOP_REQUESTED'
    if flag.exists():
        flag.unlink()
    with (CAMPAIGN / 'confirmation.log').open('a') as log:
        process = subprocess.Popen([str(PYTHON), '-u', '-m', 'ourwork.campaign_v2.confirm',
                                    '--hours', str(hours)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print('Holdout confirmation started. PID:', process.pid, '; inspect confirmation/status.json and download snapshots.')
    return process.pid
