"""Background setup for a new Kaggle T4x2 session; no model compilation."""
from pathlib import Path
import hashlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = Path('/kaggle/working/syftr_project')
OUTPUT = Path('/kaggle/working/baseline_results')
SCRATCH = Path('/kaggle/temp/rm-syftr')
STATUS = Path('/kaggle/working/v2_setup_status.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def status(stage, **extra):
    record = {'status': stage, 'updated_unix': time.time(), **extra}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(STATUS)
    print(stage, extra, flush=True)


def check_paths(names):
    for name in names:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts, name




def main():
    setup_lock = Path('/kaggle/working/.v2_setup.lock').open('a')
    fcntl.flock(setup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        assert not any(m in args for m in ['ourwork.baseline.run', 'ourwork.baseline.worker',
                   'ourwork.campaign_v2.run', 'ourwork.campaign_v2.controller', 'ourwork.campaign_v2.confirm']), 'An experiment is active'
    status('verifying_bundle')
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    check_paths(manifest)
    for name, checksum in manifest.items():
        assert sha(HERE / name) == checksum, name
    # Plotting uses Kaggle's notebook Python, not the experimental Python environment.
    subprocess.run([sys.executable, '-c', 'import matplotlib; print("PLOTTING_READY", matplotlib.__version__)'], check=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True)
    assert gpu.count('T4') == 2, 'Select GPU T4 x2 in Kaggle before setup'
    status('restoring_saved_runtime')
    if not (OUTPUT / 'runtime_checkpoint_manifest.json').exists():
        with tarfile.open(HERE / 'gpu1_checkpoint.tar.gz', 'r:gz') as t:
            members = t.getmembers()
            check_paths(m.name for m in members)
            assert all(Path(m.name).parts[0] == 'baseline_results' and not m.issym() and not m.islnk() for m in members)
            t.extractall('/kaggle/working', filter='data')
    runtime = json.loads((OUTPUT / 'runtime_checkpoint_manifest.json').read_text())
    assert sha(OUTPUT / 'server_metadata.json') == runtime['metadata_sha256']
    for name, checksum in runtime['files_sha256'].items():
        assert Path(name).name == name
        assert sha(OUTPUT / 'runtime/bin' / name) == checksum, name
    # Restore exact GPU evaluator rather than the old CPU source in historical archives.
    ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = OUTPUT / 'executed_source_gpu1'
    with zipfile.ZipFile(HERE / 'frozen_common.zip') as z:
        check_paths(z.namelist())
        source_manifest = json.loads(z.read('BUNDLE_MANIFEST.json'))
        for name, checksum in source_manifest.items():
            data = z.read(name)
            assert hashlib.sha256(data).hexdigest() == checksum, name
            for target in [ROOT / name, snapshot / name]:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f'Existing common source differs: {target}; use a clean session')
                target.write_bytes(data)
        for target in [ROOT / 'BUNDLE_MANIFEST.json', snapshot / 'BUNDLE_MANIFEST.json']:
            target.write_bytes(z.read('BUNDLE_MANIFEST.json'))
    for name in manifest:
        if name.startswith('source/'):
            rel = Path(name).relative_to('source')
            assert str(rel).startswith(('ourwork/optimizers/', 'ourwork/campaign_v2/', 'ourwork/experiment_protocols/'))
            target = ROOT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (HERE / name).read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Existing supplemental source differs: {target}')
            target.write_bytes(data)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from ourwork.campaign_v2.common import CAMPAIGN, atomic, implementation_identity
    from ourwork.campaign_v2.recovery import restore_capsule
    restore_capsule(CAMPAIGN, Path('/kaggle/input'), OUTPUT)
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    identity = implementation_identity(ROOT)
    certificate = CAMPAIGN / 'implementation.json'
    if certificate.exists():
        assert json.loads(certificate.read_text()) == identity, 'Saved campaign uses another implementation'
    atomic(certificate, identity)
    import importlib.metadata
    atomic(CAMPAIGN / 'plotting_python.json', {'executable': sys.executable,
        'python_version': sys.version, 'matplotlib_version': importlib.metadata.version('matplotlib')})
    profile = json.loads((ROOT / 'ourwork/campaign_v2/protocol.json').read_text())
    selected = CAMPAIGN / 'protocol.json'
    if selected.exists():
        assert json.loads(selected.read_text()) == profile, 'Saved campaign protocol differs'
    atomic(selected, profile)
    atomic(CAMPAIGN / 'design.json', {
        'version': 'restricted-v2-25x10', 'methods': ['syftr_motpe', 'nsga2', 'hybrid_motpe', 'adaptive_nsga2'],
        'seeds': [11], 'attempts_per_arm': 25, 'search_qa_count': 10, 'startup': 10,
        'population': 5, 'evolutionary_generations': 3, 'intermediate_checkpoints': [10,15,20,25],
        'schedule': 'Sequential round-robin: every arm reaches the same planned checkpoint before the next round.',
        'amendment': 'Independent restricted-space backup study; do not pool with full-space results.',
        'cache_policy': 'Shared pinned input/index artifacts; each method re-evaluates its own configurations; no answer or fitness reuse.',
        'timing_limit': 'Warm shared cache and fixed arm order; descriptive timings only.',
        'confirmation': {'qa_count': 10, 'selection': 'max_single_point_hv_then_f1_then_lower_tokens_then_earliest_trial'},
        'scope': 'Exploratory single-seed local-model adaptation; no guaranteed superiority or across-seed significance.'})
    archived_code = CAMPAIGN / 'supplemental_source'
    for rel, checksum in identity['files'].items():
        target = archived_code / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    atomic(archived_code / 'MANIFEST.json', identity)
    status('restoring_locked_python')
    subprocess.run([sys.executable, '-u', '-m', 'ourwork.baseline.bootstrap', '--root', str(ROOT),
                    '--scratch', str(SCRATCH)], cwd=ROOT, check=True)
    python = SCRATCH / 'venv/bin/python'
    status('software_validation')
    test_command = [str(python), '-m', 'unittest', 'ourwork.baseline.test_baseline',
                    'ourwork.baseline.test_device_policy', 'ourwork.optimizers.test_optimizers',
                    'ourwork.optimizers.test_bridge', 'ourwork.optimizers.test_integration',
                    'ourwork.campaign_v2.test_campaign', 'ourwork.campaign_v2.test_restricted', '-q']
    subprocess.run(test_command, cwd=ROOT, check=True)
    subprocess.run([str(python), '-c',
        'import json; from pathlib import Path; from ourwork.baseline.records import source_identity; '
        'cap=json.loads(Path("/kaggle/working/baseline_results/capabilities-f1-gpu1-seed11/protocol.json").read_text()); '
        'from ourwork.campaign_v2.common import checked_source_identity; '
        'from ourwork.baseline.records import fingerprint; import hashlib; '
        'expected=json.loads(json.dumps(cap["source_identity"])); '
        'amendment=json.loads(Path("ourwork/campaign_v2/restricted_source_amendment.json").read_text()); '
        'assert hashlib.sha256(Path("ourwork/baseline/spec.py").read_bytes()).hexdigest()==amendment["restricted_file_sha256"]; '
        'expected["files"]["ourwork/baseline/spec.py"]=amendment["restricted_file_sha256"]; '
        'expected["sha256"]=fingerprint(expected["files"]); '
        'checked_source_identity(source_identity(Path.cwd()), expected); '
        'print("COMMON_EVALUATOR_CONTENT_IDENTITY_VERIFIED")'], cwd=ROOT, check=True)
    status('restoring_pinned_weights')
    downloader = '''from pathlib import Path
import hashlib,json
from huggingface_hub import hf_hub_download
meta=json.loads(Path('/kaggle/working/baseline_results/server_metadata.json').read_text())
paths={}
for name, expected in meta['files_sha256'].items():
    p=Path(hf_hub_download(meta['model_repo'], filename=name, revision=meta['model_revision'],
                          cache_dir='/kaggle/temp/rm-syftr/server/weights'))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    assert h.hexdigest()==expected,name
    paths[name]=str(p)
    print('WEIGHT_VERIFIED',name,flush=True)
original=meta['command'][meta['command'].index('-m')+1]
assert paths[Path(original).name]==original
'''
    subprocess.run([str(python), '-u', '-c', downloader], cwd=ROOT, check=True)
    metadata = json.loads((OUTPUT / 'server_metadata.json').read_text())
    binary = Path(metadata['command'][0])
    assert binary.is_relative_to(SCRATCH), 'Unexpected original runtime location'
    binary.parent.mkdir(parents=True, exist_ok=True)
    for name, checksum in runtime['files_sha256'].items():
        target = binary.parent / name
        if not target.exists() or sha(target) != checksum:
            shutil.copy2(OUTPUT / 'runtime/bin' / name, target)
        assert sha(target) == checksum
    binary.chmod(binary.stat().st_mode | 0o111)
    assert sha(binary) == metadata['binary_sha256']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    dirs = [str(binary.parent), '/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu',
            '/usr/lib64-nvidia', '/usr/local/cuda/compat']
    env['LD_LIBRARY_PATH'] = ':'.join(dirs + [env.get('LD_LIBRARY_PATH', '')])
    status('starting_original_gpu0_server')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3) as response:
            ready = response.status == 200
    except OSError:
        ready = False
    if ready:
        marker = CAMPAIGN / 'server_process.json'
        assert marker.exists(), 'An unrecognized server already occupies port 8765'
        saved_pid = json.loads(marker.read_text())['pid']
        actual_command = Path(f'/proc/{saved_pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        assert actual_command == metadata['command'], 'Server process identity differs from the saved command'
    else:
        with (OUTPUT / 'v2_server.log').open('a') as log:
            server = subprocess.Popen(metadata['command'], cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        atomic(CAMPAIGN / 'server_process.json', {'pid': server.pid, 'binary_sha256': sha(binary)})
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError('Original model server failed; inspect v2_server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3) as response:
                    ready = response.status == 200
                if ready:
                    break
            except OSError:
                pass
            time.sleep(2)
        assert ready, 'Server did not become ready within six minutes'
    status('gpu_preflight')
    subprocess.run([str(python), '-m', 'ourwork.baseline.preflight', '--output',
                    str(OUTPUT / 'preflight_gpu1_session.json')], cwd=ROOT, check=True)
    embedding_check = '''import json
from pathlib import Path
from ourwork.baseline.adapter import prepare
from ourwork.baseline.device_policy import device_diagnostics
from ourwork.campaign_v2.common import workspace_for
out=Path('/kaggle/working/baseline_results')
protocol=json.loads(Path('ourwork/campaign_v2/protocol.json').read_text())
sources=json.loads((out/'capabilities-f1-gpu1-seed11/protocol.json').read_text())['sources']
study,_,hf=prepare(protocol, workspace_for('syftr_motpe',11), sources)
model=hf.get_hf_embedding_model('BAAI/bge-small-en-v1.5',timeout_config=study.timeouts)
v=model.get_text_embedding('Research embedding readiness.')
assert len(v)==384 and all(__import__('math').isfinite(x) for x in v)
record={'status':'passed','device':device_diagnostics(protocol),'dimensions':len(v)}
(out/'campaign-v2/fresh_embedding_readiness.json').write_text(json.dumps(record,indent=2))
print('GPU1_EMBEDDING_READY',record,flush=True)
'''
    subprocess.run([str(python), '-u', '-c', embedding_check], cwd=ROOT, check=True)
    atomic(OUTPUT / 'active_source.json', {'source_directory': 'executed_source_gpu1',
        'supplemental_source_directory': 'campaign-v2/supplemental_source',
        'restore_policy': 'Restore exact common source plus matching v2 supplemental source.'})
    provenance = CAMPAIGN / 'readiness'
    provenance.mkdir(exist_ok=True)
    for name in ['server_metadata.json','preflight_gpu1_session.json','runtime_checkpoint_manifest.json']:
        shutil.copy2(OUTPUT / name, provenance / name)
    setup_log = Path('/kaggle/working/v2_setup.log')
    if setup_log.exists():
        shutil.copy2(setup_log, provenance / 'setup.log')
    atomic(provenance / 'software_validation.json', {'status':'passed','expected_test_count':54,
        'scope':'Software validation with synthetic workers; real campaign runs still to execute.'})
    status('fresh_restricted_capabilities')
    subprocess.run([str(python), '-u', '-m', 'ourwork.campaign_v2.restricted_capabilities'], cwd=ROOT, check=True)
    from ourwork.campaign_v2.snapshots import snapshot as save_snapshot
    save_snapshot(CAMPAIGN)
    status('ready', campaign='restricted-v2-25x10', actual_comparative_results_available=False,
           note='Software and fresh allocation readiness checks passed. Call launch(hours=8).')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('error', error=f'{type(exc).__name__}: {exc}', log='/kaggle/working/v2_setup.log')
        raise
