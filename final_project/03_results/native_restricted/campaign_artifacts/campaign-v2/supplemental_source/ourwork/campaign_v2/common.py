from pathlib import Path
import hashlib
import json
import os

ROOT = Path('/kaggle/working/syftr_project')
OUTPUT = Path('/kaggle/working/baseline_results')
CAMPAIGN = OUTPUT / 'campaign-v2'
SCRATCH = Path('/kaggle/temp/rm-syftr')
PYTHON = SCRATCH / 'venv/bin/python'
METHODS = ['syftr_motpe', 'nsga2', 'hybrid_motpe', 'adaptive_nsga2']
SEEDS = [11]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def implementation_identity(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    files = {}
    for directory in ['ourwork/campaign_v2', 'ourwork/optimizers']:
        for p in sorted((root / directory).iterdir()):
            if p.suffix in {'.py', '.json'}:
                files[str(p.relative_to(root))] = digest(p)
    encoded = json.dumps(files, sort_keys=True).encode()
    return {'files': files, 'sha256': hashlib.sha256(encoded).hexdigest()}


def workspace_for(method, seed, scratch=SCRATCH):
    """Only retrieval artifacts/downloads are shared; answers/fitness are never cached."""
    scratch = Path(scratch)
    workspace = scratch / 'campaign-v2' / f'{method}-seed{seed}'
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ['indexes', 'huggingface_cache', 'hf-home']:
        shared = scratch / 'campaign-v2-shared' / name
        shared.mkdir(parents=True, exist_ok=True)
        target = workspace / name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != shared.resolve():
                raise ValueError(f'Unexpected cache path: {target}')
        else:
            target.symlink_to(shared, target_is_directory=True)
    return workspace


def checked_source_identity(actual, expected):
    """Require exact code bytes; tolerate only Git's non-repository error wording.

    Git prints a mount-boundary-dependent error on these archive-based checkouts.
    The actual observation is separately archived by the caller. No successful Git
    revision, file hash, missing file, or other provenance difference is ignored.
    """
    if actual == expected:
        return expected
    if set(actual) != set(expected) or any(actual[k] != expected[k] for k in actual if k != 'git_head'):
        raise ValueError('Common evaluator source content/identity differs from reviewed capability')
    for item in [actual.get('git_head', {}), expected.get('git_head', {})]:
        if (item.get('returncode') != 128 or item.get('stdout') != '' or
                not item.get('stderr', '').startswith('fatal: not a git repository')):
            raise ValueError('Git revision identity changed; only non-repository diagnostic text may differ')
    return expected
