"""Consistent study capsules; the attached base checkpoint supplies runtime binaries."""
from pathlib import Path
import json
import hashlib
import sqlite3
import tempfile
import time
import zipfile
from .common import CAMPAIGN, atomic, digest


def snapshot(campaign=CAMPAIGN, export_dir=None):
    campaign = Path(campaign)
    export_dir = Path(export_dir or '/kaggle/working/campaign_v2_exports')
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / 'campaign_v2_review.zip'
    temporary = target.with_suffix('.zip.tmp')
    inventory = {}
    with tempfile.TemporaryDirectory(prefix='syftr-db-') as tmp:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in sorted(campaign.rglob('*')):
                if not p.is_file() or p.is_symlink() or p.name.endswith(('-wal', '-shm', '.tmp', '.lock')):
                    continue
                rel = p.relative_to(campaign)
                source = p
                if p.suffix == '.sqlite3':
                    source = Path(tmp) / str(len(inventory))
                    with sqlite3.connect(f'file:{p}?mode=ro', uri=True) as live:
                        with sqlite3.connect(source) as backup:
                            live.backup(backup)
                archive_name = str(Path('campaign-v2') / rel)
                # Read once: controller heartbeat files may be atomically replaced.
                data = source.read_bytes()
                z.writestr(archive_name, data)
                inventory[archive_name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            z.writestr('CAPSULE_MANIFEST.json', json.dumps(inventory, indent=2))
        with zipfile.ZipFile(temporary) as z:
            assert z.testzip() is None
    temporary.replace(target)
    receipt = {'file': target.name, 'bytes': target.stat().st_size,
               'sha256': digest(target), 'saved_unix': time.time(),
               'scope': 'Campaign raw records, consistent SQLite snapshots, sampler state and versioned metadata. '
                        'Restore with the attached v2 source/base-runtime package; no scratch indexes/weights.',
               'files': len(inventory)}
    atomic(export_dir / 'campaign_v2_receipt.json', receipt)
    return receipt
