"""Kaggle environment setup, with all large caches in disposable scratch storage."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--scratch', default='/kaggle/temp/rm-syftr')
    args = parser.parse_args()
    root, scratch = Path(args.root).resolve(), Path(args.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(UV_CACHE_DIR=str(scratch / 'uv-cache'), UV_PROJECT_ENVIRONMENT=str(scratch / 'venv'),
               UV_PYTHON_INSTALL_DIR=str(scratch / 'python'), TMPDIR=str(scratch / 'tmp'),
               PIP_CACHE_DIR=str(scratch / 'pip-cache'), HF_HOME=str(scratch / 'hf-home'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    # Do not replace the notebook's global uv executable or Python packages.
    uv_target = scratch / 'bootstrap-packages'
    env['PYTHONPATH'] = str(uv_target)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', str(uv_target),
                    'uv==0.9.6'], env=env, check=True)
    # Uses the supplied lock, not the changing Kaggle preinstalled package mix.
    subprocess.run([sys.executable, '-m', 'uv', 'sync', '--frozen', '--no-dev', '--python', '3.12'],
                   cwd=root, env=env, check=True)
    python = scratch / 'venv/bin/python'
    subprocess.run([str(python), '-m', 'pip', 'check'], cwd=root, env=env, check=True)
    # Installation/import validation is explicit and separate from unit tests.
    subprocess.run([str(python), '-c',
                    'import torch, optuna, ray; from syftr.tuner.qa_tuner import build_flow; '
                    'print("SYFTR_IMPORT_OK", torch.__version__, optuna.__version__, ray.__version__)'],
                   cwd=root, env=env, check=True)
    print('BASELINE_PYTHON=' + str(python), flush=True)


if __name__ == '__main__':
    main()
