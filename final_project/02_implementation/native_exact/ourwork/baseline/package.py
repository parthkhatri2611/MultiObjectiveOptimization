"""Allowlisted source bundle and a small, reproducible Kaggle notebook."""
import hashlib
import json
from pathlib import Path
import zipfile


def package():
    root = Path(__file__).resolve().parents[2]
    out = Path(__file__).parent / 'dist'
    out.mkdir(exist_ok=True)
    selected = [root / name for name in ['pyproject.toml', 'uv.lock', 'README.md', 'LICENSE']]
    for base in [root / 'syftr', root / 'ourwork/baseline']:
        selected.extend(p for p in base.rglob('*') if p.is_file()
                        and not any(x in p.parts for x in ['.venv', 'dist', 'work', 'runs', '__pycache__'])
                        and p.suffix in {'.py', '.json', '.md', '.yaml'})
    # Never include .git, config.yaml, credentials, runtime environments or previous result datasets.
    archive = out / 'syftr_baseline_source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(selected)):
            z.write(p, p.relative_to(root))
        z.writestr('BUNDLE_MANIFEST.json', json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sorted(set(selected))}, sort_keys=True, indent=2))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / 'SHA256SUMS.txt').write_text(sha + '  ' + archive.name + '\n')

    def md(text):
        return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}
    def code(text):
        return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                'source': text.splitlines(keepends=True)}
    cells = [md('''# Syftr local-model baseline

Qwen2.5-7B-Instruct Q4_K_M; prepared HotPotQA train_hard; original five-family Syftr hierarchy.
This notebook bootstraps normal Python source. No performance results are prefilled.
Attach `syftr_baseline_source.zip` as a private input. Select GPU T4 x2 and enable Internet.
Default execution runs **smoke** only. Run capabilities and pilot before baseline.
Keep all generated research outputs private until reviewed.
'''), code(f'''from pathlib import Path
import hashlib, json, os, subprocess, sys, time, urllib.request, zipfile
SOURCE_SHA256 = {sha!r}
matches = list(Path('/kaggle/input').rglob('syftr_baseline_source.zip'))
ROOT = Path('/kaggle/working/syftr_project')
ROOT.mkdir(parents=True, exist_ok=True)
if matches:
    assert len(matches) == 1
    assert hashlib.sha256(matches[0].read_bytes()).hexdigest() == SOURCE_SHA256, 'Source checksum mismatch'
    with zipfile.ZipFile(matches[0]) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(ROOT)
else:
    import shutil
    manifests = list(Path('/kaggle/input').rglob('BUNDLE_MANIFEST.json'))
    assert len(manifests) == 1, 'Attach the final source input with BUNDLE_MANIFEST.json'
    shutil.copytree(manifests[0].parent, ROOT, dirs_exist_ok=True)
for name, digest in json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
SCRATCH = Path('/kaggle/temp/rm-syftr')
OUTPUT = Path('/kaggle/working/baseline_results')
OUTPUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print('Verified source copied. Blocking stage execution follows.')
'''), code('''# Complete blocking run, suitable for Save & Run All.
# Advance STAGE only after reviewing the preceding stage's outputs.
STAGE = 'smoke'  # smoke -> capabilities -> pilot -> baseline
SEED = 11
RESUME = False
command = [sys.executable, '-m', 'ourwork.baseline.session', '--stage', STAGE,
           '--seed', str(SEED), '--scratch', str(SCRATCH), '--output', str(OUTPUT)]
if RESUME: command.append('--resume')
subprocess.run(command, cwd=ROOT, check=True)
run_dir = OUTPUT / f'{STAGE}-seed{SEED}'
print((run_dir / 'analysis/summary.json').read_text())
print((run_dir / 'analysis/FINDINGS.md').read_text())
'''), md('''## Preserve results

Download or save the completed `baseline_results` outputs before stopping the session.
For cross-session resume, attach the saved output as input and copy it into the same writable output location.
Restore the exact source/server/package versions; the runner rejects changed protocols.
Do not assume draft files or background processes survive notebook termination.
Save & Run All starts a clean execution; this notebook uses the blocking session entry point.
Earlier stage outputs must be restored before progressing to pilot or baseline.
''')]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                            'language_info': {'name': 'python', 'version': '3.12'}},
                'nbformat': 4, 'nbformat_minor': 5}
    (out / 'Syftr_Kaggle_Baseline.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    print(json.dumps({'archive': str(archive), 'sha256': sha, 'bytes': archive.stat().st_size,
                      'notebook': str(out / 'Syftr_Kaggle_Baseline.ipynb')}, indent=2))


if __name__ == '__main__':
    package()
