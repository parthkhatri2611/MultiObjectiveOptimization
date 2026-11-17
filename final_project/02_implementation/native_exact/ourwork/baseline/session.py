"""Blocking Kaggle batch entry point; waits and fails explicitly instead of ending early."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'capabilities', 'pilot', 'baseline'], default='smoke')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--scratch', default='/kaggle/temp/rm-syftr')
    parser.add_argument('--output', default='/kaggle/working/baseline_results')
    parser.add_argument('--skip-setup', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    scratch, output = Path(args.scratch).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = scratch / 'venv/bin/python'
    if not args.skip_setup:
        with (output / 'environment_setup.log').open('a') as log:
            subprocess.run([sys.executable, '-m', 'ourwork.baseline.bootstrap', '--root', str(root),
                            '--scratch', str(scratch)], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(python), '-m', 'unittest', 'ourwork.baseline.test_baseline'], cwd=root, check=True)
    with (output / 'server.log').open('a') as log:
        server = subprocess.Popen([str(python), '-m', 'ourwork.baseline.serve', '--workspace', str(scratch / 'server'),
                                   '--metadata', str(output / 'server_metadata.json')], cwd=root,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ready = False
            for _ in range(240):
                if server.poll() is not None:
                    raise RuntimeError('Model server failed; inspect server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3) as response:
                        ready = response.status == 200
                    if ready:
                        break
                except OSError:
                    time.sleep(10)
            if not ready:
                raise TimeoutError('Model server not ready within 40 minutes')
            subprocess.run([str(python), '-m', 'ourwork.baseline.preflight', '--output', str(output / 'preflight.json')],
                           cwd=root, check=True)
            run_dir = output / f'{args.stage}-seed{args.seed}'
            command = [str(python), '-m', 'ourwork.baseline.run', '--protocol',
                       str(root / f'ourwork/baseline/configs/{args.stage}.json'), '--output', str(run_dir),
                       '--workspace', str(scratch / 'evaluation'), '--seed', str(args.seed), '--max-hours', '9',
                       '--server-metadata', str(output / 'server_metadata.json'), '--preflight', str(output / 'preflight.json')]
            if args.stage in {'pilot', 'baseline'}:
                command += ['--capabilities-run', str(output / 'capabilities-seed11')]
            if args.resume:
                command += ['--resume']
            with (output / f'{args.stage}-seed{args.seed}.log').open('a') as run_log:
                subprocess.run(command, cwd=root, stdout=run_log, stderr=subprocess.STDOUT, check=True)
            print((run_dir / 'analysis/summary.json').read_text())
        finally:
            import signal
            os.killpg(server.pid, signal.SIGTERM) if server.poll() is None else None
            server.wait(timeout=30)


if __name__ == '__main__':
    main()
