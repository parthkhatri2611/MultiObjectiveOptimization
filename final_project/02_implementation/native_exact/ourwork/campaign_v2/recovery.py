"""Verify and restore a downloaded result capsule, ZIP or Kaggle-expanded input."""
from pathlib import Path
import hashlib
import json
import zipfile


def check_paths(names):
    for name in names:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts, name


def restore_capsule(campaign, input_root, output):
    campaign, input_root, output = Path(campaign), Path(input_root), Path(output)
    archives = list(input_root.rglob('campaign_v2_review.zip'))
    expanded = list(input_root.rglob('CAPSULE_MANIFEST.json'))
    if not archives and not expanded:
        return
    if (campaign / 'runs').exists():
        print('Existing campaign results retained; input capsule not overlaid.', flush=True)
        return
    from contextlib import ExitStack
    with ExitStack() as stack:
        if archives:
            assert len(archives) == 1, 'Attach only the most recent campaign_v2_review.zip'
            z = stack.enter_context(zipfile.ZipFile(archives[0]))
            check_paths(z.namelist())
            manifest = json.loads(z.read('CAPSULE_MANIFEST.json'))
            assert set(z.namelist()) == set(manifest) | {'CAPSULE_MANIFEST.json'}
            read_data = z.read
        else:
            assert len(expanded) == 1, 'Attach only one automatically expanded capsule'
            manifest = json.loads(expanded[0].read_text())
            capsule_root = expanded[0].parent
            def read_data(name):
                source = capsule_root / name
                assert not source.is_symlink(), 'Capsule files must be ordinary files'
                return source.read_bytes()
        check_paths(manifest)
        for name, item in manifest.items():
            assert Path(name).parts[0] == 'campaign-v2'
            data = read_data(name)
            assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], name
        for name in manifest:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_data(name))
    print('Verified saved campaign capsule restored.', flush=True)

