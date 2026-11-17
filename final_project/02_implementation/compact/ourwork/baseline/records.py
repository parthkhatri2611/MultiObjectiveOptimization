"""Crash-conscious journals, immutable protocol identity and experiment provenance."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(canonical(value) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith(b'\n'):
                break  # A torn trailing write is recoverable; earlier corruption is not.
            raise
    return result


def repair_trailing_record(path):
    path = Path(path)
    if path.exists() and (data := path.read_bytes()) and not data.endswith(b'\n'):
        # Preserve the interrupted bytes before truncating an incomplete record.
        try:
            json.loads(data.splitlines()[-1])
        except json.JSONDecodeError:
            path.with_suffix(path.suffix + f'.torn-{time.time_ns()}').write_bytes(data)
            path.write_bytes(data[:data.rfind(b'\n') + 1])
        else:
            with path.open('ab') as f:
                f.write(b'\n')


def command(args, cwd=None):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'error': str(e)}


def source_identity(root):
    root = Path(root)
    files = {}
    for base in [root / 'syftr', root / 'ourwork' / 'baseline']:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ['.venv', 'runs', 'work', 'dist', '__pycache__']):
                continue
            if p.suffix in {'.py', '.json', '.yaml', '.toml', '.txt', '.sh'}:
                files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['pyproject.toml', 'uv.lock']:
        p = root / name
        files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {'files': files, 'sha256': fingerprint(files),
            'git_head': command(['git', 'rev-parse', 'HEAD'], root)}


def environment():
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'cpu_count': os.cpu_count(),
            'memory': Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,
            'gpu': command(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv']),
            'packages': command([sys.executable, '-m', 'pip', 'freeze']),
            'disk': command(['df', '-h', '.', '/tmp']),
            'cuda': command(['nvcc', '--version'])}


class RunStore:
    def __init__(self, path, protocol, resume=False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity = fingerprint(protocol)
        manifest = self.path / 'protocol.json'
        if manifest.exists():
            if not resume:
                raise FileExistsError('Run exists; explicitly request --resume')
            if fingerprint(json.loads(manifest.read_text())) != self.identity:
                raise ValueError('Resume protocol differs from the saved protocol')
        else:
            if resume:
                raise FileNotFoundError('Cannot resume a run without protocol.json')
            atomic_json(manifest, protocol)
        for journal in self.path.rglob('*.jsonl'):
            repair_trailing_record(journal)

    def event(self, kind, **data):
        append_jsonl(self.path / 'events.jsonl', {'time': time.time(), 'kind': kind, **data})
