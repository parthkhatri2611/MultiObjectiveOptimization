"""Notebook-facing serial execution with progress and automatic recovery exports.

Runs under the notebook's launcher Python; actual research uses pinned Python 3.12.
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path('/kaggle/working/syftr_project')
OUTPUT = Path('/kaggle/working/baseline_results')
SCRATCH = Path('/kaggle/temp/rm-syftr')
PYTHON = str(SCRATCH / 'venv/bin/python')
METHODS = ['nsga2', 'hybrid_motpe', 'adaptive_nsga2']


def export_results(label):
    from IPython.display import FileLink, display
    tag = f'{label}_{time.time_ns()}'
    review = Path('/kaggle/working') / f'{tag}_review.zip'
    checkpoint = Path('/kaggle/working') / f'{tag}_checkpoint.tar.gz'
    with zipfile.ZipFile(review, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and 'runtime' not in p.relative_to(OUTPUT).parts:
                z.write(p, p.relative_to(OUTPUT))
    with tarfile.open(checkpoint, 'w:gz', compresslevel=1) as tar:
        tar.add(OUTPUT, arcname='baseline_results')
    for path in [review, checkpoint]:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        print(path.name, 'SHA256', h.hexdigest(), flush=True)
        display(FileLink(str(path)))
    return review, checkpoint


def run_method(method, seed=11, max_hours=4):
    if method not in METHODS + ['syftr_motpe'] or seed not in {11, 23, 37}:
        raise ValueError('Use a declared method and seed')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(m in args for m in ['ourwork.baseline.run', 'ourwork.baseline.worker', 'ourwork.optimizers.run']):
            raise RuntimeError('An experiment is already running; use serial execution')
    with urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=10) as response:
        assert response.status == 200
    if method != 'syftr_motpe':
        install = json.loads((OUTPUT / 'optimizer_installation.json').read_text())
        assert install['status'] == 'passed'
        actual = json.loads(subprocess.check_output([
            PYTHON, '-c', 'import json; from ourwork.optimizers.run import implementation_identity; '
            'print(json.dumps(implementation_identity()))'
        ], cwd=ROOT, text=True))
        assert actual == install['optimizer_identity'], 'Optimizer changed after validation'
    label = ('baseline' if method == 'syftr_motpe' else method) + f'-f1-gpu1-seed{seed}'
    run = OUTPUT / label
    workspace = SCRATCH / label
    resume = (run / 'protocol.json').exists()
    workspace.mkdir(parents=True, exist_ok=True)
    if not resume and any((workspace / 'indexes').rglob('*')):
        raise ValueError('A new run must start with its own empty index cache')
    for name in ['huggingface_cache', 'hf-home']:
        shared = SCRATCH / 'evaluation' / name
        shared.mkdir(parents=True, exist_ok=True)
        target = workspace / name
        if target.exists() or target.is_symlink():
            assert target.is_symlink() and target.resolve() == shared.resolve()
        else:
            target.symlink_to(shared, target_is_directory=True)
    command = [PYTHON, '-u', '-m', 'ourwork.baseline.run' if method == 'syftr_motpe' else 'ourwork.optimizers.run']
    if method == 'syftr_motpe':
        profile = json.loads((ROOT / 'ourwork/baseline/configs/baseline_f1_tokens.json').read_text())
        profile.update(embedding_device='cuda:1', cache_policy='isolated-per-run-indexes-v1')
        profile_path = OUTPUT / 'selected_protocols/baseline_f1_gpu1_v1.json'
        if profile_path.exists():
            assert json.loads(profile_path.read_text()) == profile
        else:
            profile_path.parent.mkdir(exist_ok=True)
            profile_path.write_text(json.dumps(profile, indent=2) + '\n')
        command += ['--protocol', str(profile_path)]
    else:
        reference = OUTPUT / f'baseline-f1-gpu1-seed{seed}'
        s = json.loads((reference / 'analysis/summary.json').read_text())
        assert s['budget_exhausted'], 'Finish the matching baseline before running proposed methods'
        command += ['--method', method, '--baseline-run', str(reference)]
    command += ['--output', str(run), '--workspace', str(workspace), '--seed', str(seed),
                '--max-hours', str(max_hours), '--server-metadata', str(OUTPUT / 'server_metadata.json'),
                '--preflight', str(OUTPUT / 'preflight_gpu1_session.json'),
                '--capabilities-run', str(OUTPUT / 'capabilities-f1-gpu1-seed11')]
    if resume:
        command.append('--resume')
    run.mkdir(exist_ok=True)
    (run / 'execution_command.json').write_text(json.dumps({
        'command': command, 'method': method, 'seed': seed,
        'cache_policy': 'isolated indexes; shared pinned downloads; scratch may be rebuilt after restart'
    }, indent=2) + '\n')
    log_path = OUTPUT / (label + '.log')
    print('RUNNING', label, '30 attempts x 20 questions; GPU0 generator / GPU1 embeddings.', flush=True)
    try:
        with log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        exit_code = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        summary = run / 'analysis/summary.json'
                        if summary.exists():
                            s = json.loads(summary.read_text())
                            print('PROGRESS', {k: s[k] for k in ['attempts', 'budget', 'completed', 'failed']}, flush=True)
                        print('CONTROLLER:', log_path.read_text(errors='replace')[-500:], flush=True)
                        candidates = sorted((run / 'candidates').glob('*'))
                        if candidates:
                            p = candidates[-1] / 'progress.json'
                            if p.exists():
                                s = json.loads(p.read_text())
                                print('ACTIVE', candidates[-1].name, 'QA', s['questions_attempted'], '/', s['questions_expected'], flush=True)
                            p = candidates[-1] / 'worker.log'
                            if p.exists():
                                print('WORKER:', '\n'.join(p.read_text(errors='replace').replace('\r', '\n').splitlines()[-4:]), flush=True)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():
                        continue
                    try:
                        args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
                        if 'ourwork.baseline.worker' in args and any(a.startswith(str(run) + '/') for a in args):
                            os.kill(int(proc.name), signal.SIGTERM)
                    except OSError:
                        pass
                raise
        print(log_path.read_text(errors='replace')[-2500:])
        if exit_code:
            raise RuntimeError(f'Experiment exited {exit_code}; review logs')
        summary = json.loads((run / 'analysis/summary.json').read_text())
        print('FINAL SUMMARY', json.dumps(summary, indent=2))
        return summary
    finally:
        export_results(label)


def run_all_methods(seed=11, max_hours=4):
    """Sequential convenience only; stops if a budget is incomplete or a stage fails."""
    orders = {11: METHODS, 23: ['adaptive_nsga2', 'nsga2', 'hybrid_motpe'],
              37: ['hybrid_motpe', 'adaptive_nsga2', 'nsga2']}
    for method in orders[seed]:
        result = run_method(method, seed, max_hours)
        if not result['budget_exhausted']:
            raise RuntimeError('Resume the incomplete run before proceeding to the next method')
