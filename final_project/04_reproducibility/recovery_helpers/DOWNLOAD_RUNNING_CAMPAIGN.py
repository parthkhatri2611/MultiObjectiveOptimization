# Run once in a new cell in the EXISTING friend/restricted Kaggle session.
# Reads a completed export; does not stop the controller or edit experiment state.
from pathlib import Path
from IPython.display import HTML, display
import base64, hashlib, io, json, time, zipfile

source = Path('/kaggle/working/campaign_v2_exports/campaign_v2_review.zip')
assert source.exists(), 'No completed export found. Keep the session open and share this message.'
data = source.read_bytes()  # Export is replaced atomically; this captures one completed version.
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    assert archive.testzip() is None, 'ZIP integrity failed; keep session open.'
    inventory = json.loads(archive.read('CAPSULE_MANIFEST.json'))
    for name, entry in inventory.items():
        contents = archive.read(name)
        assert len(contents) == entry['bytes'], name
        assert hashlib.sha256(contents).hexdigest() == entry['sha256'], name

stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
name = f'restricted_progress_{stamp}.zip'
target = Path('/kaggle/working') / name
target.write_bytes(data)
receipt = {'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
           'copied_unix': time.time(), 'source': str(source), 'verified_files': len(inventory),
           'scope': 'Immutable copy of latest completed export. Active/incomplete candidate may be absent.'}
receipt_name = target.with_suffix('.receipt.json').name
receipt_bytes = (json.dumps(receipt, indent=2) + '\n').encode()
target.with_suffix('.receipt.json').write_bytes(receipt_bytes)
print(json.dumps(receipt, indent=2))
print('Click BOTH download links below. They bypass the Output file tree.')
for filename, content, mime in [(name, data, 'application/zip'),
                                (receipt_name, receipt_bytes, 'application/json')]:
    encoded = base64.b64encode(content).decode('ascii')
    display(HTML(f'<p><a download="{filename}" href="data:{mime};base64,{encoded}">'
                 f'Download {filename}</a></p>'))
print('Keep the running session open. Return the ZIP and receipt after downloading.')
