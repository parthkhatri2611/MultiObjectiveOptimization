"""A background process, never a notebook cell waiting on model inference."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .common import CAMPAIGN, OUTPUT, METHODS, PYTHON, ROOT, atomic, read, workspace_for
from .snapshots import snapshot


def control(campaign=CAMPAIGN, hours=8):
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    lock = (campaign / '.controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profile = campaign / 'protocol.json'
    immutable = read(profile)
    deadline = time.monotonic() + hours * 3600
    status = {'status': 'running', 'pid': __import__('os').getpid(), 'started_unix': time.time(),
              'method_order': METHODS, 'seed': 11, 'protocol': immutable}
    atomic(campaign / 'status.json', status)
    def update_report(target):
        from .report import compare
        compare(campaign, make_plots=False, prefix=target)
        plotting_python = read(campaign / 'plotting_python.json')['executable']
        with (campaign / 'plotting.log').open('a') as log:
            plot_result = subprocess.run([plotting_python, '-m', 'ourwork.campaign_v2.report',
                '--plots-only', '--prefix', str(target)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status['plots_available'] = plot_result.returncode == 0
        status['latest_matched_checkpoint'] = target
        if plot_result.returncode:
            status['plotting_error'] = 'Numeric comparison passed; inspect plotting.log.'

    try:
        rounds = immutable.get('intermediate_checkpoints', [10, 15, 20, immutable['budget']])
        rounds = sorted(set(n for n in rounds if n <= immutable['budget']))
        schedule = [(target, method) for target in rounds for method in METHODS]
        for target, method in schedule:
            if (campaign / 'STOP_REQUESTED').exists() or deadline - time.monotonic() < 150:
                status['status'] = 'paused'
                break
            run = campaign / 'runs' / f'{method}-seed11'
            summary = run / 'analysis/summary.json'
            if summary.exists() and read(summary)['attempts'] >= target:
                continue
            workspace = workspace_for(method, 11)
            command = [str(PYTHON), '-u', '-m', 'ourwork.campaign_v2.run', '--method', method,
                       '--campaign', str(campaign), '--protocol', str(profile), '--output', str(run),
                       '--workspace', str(workspace), '--seed', '11', '--target-attempts', str(target),
                       '--max-hours', str((deadline - time.monotonic()) / 3600),
                       '--server-metadata', str(OUTPUT / 'server_metadata.json'),
                       '--preflight', str(OUTPUT / 'preflight_gpu1_session.json'),
                       '--capabilities-run', str(OUTPUT / 'capabilities-restricted-v2-seed11'),
                       '--export-dir', '/kaggle/working/campaign_v2_exports']
            if method != 'syftr_motpe':
                command += ['--baseline-run', str(campaign / 'runs/syftr_motpe-seed11')]
            run.mkdir(parents=True, exist_ok=True)
            atomic(run / 'execution_command.json', command)
            with (campaign / f'{method}.log').open('a') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                status.update(active_method=method, arm_pid=process.pid, active_run=str(run), planned_checkpoint=target)
                while True:
                    status['heartbeat_unix'] = time.time()
                    if summary.exists():
                        status['progress'] = read(summary)
                    atomic(campaign / 'status.json', status)
                    try:
                        code = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            achieved = read(summary)['attempts'] if summary.exists() else 0
            if code == 10 and achieved < target:
                status['status'] = 'paused'
                break
            if code not in {0, 10}:
                raise RuntimeError(f'{method} exited {code}; inspect {method}.log')
            if achieved < target:
                raise RuntimeError('Arm exited without its planned checkpoint')
            if method == METHODS[-1] and target < immutable['budget']:
                update_report(target)
        else:
            update_report(immutable['budget'])
            status['status'] = 'search_complete'
        status['finished_unix'] = time.time()
    except BaseException as e:
        status.update(status='error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        raise
    finally:
        atomic(campaign / 'status.json', status)
        snapshot(campaign)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error('Hours must be positive')
    control(hours=args.hours)
