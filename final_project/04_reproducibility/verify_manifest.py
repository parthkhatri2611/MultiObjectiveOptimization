"""Verify every immutable packaged file with SHA256, using only Python standard library."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'MANIFEST.json').read_text())
for n,v in m['files'].items():
 h=hashlib.sha256();p=root/n
 assert p.is_file() and p.stat().st_size==v['bytes'],n
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==v['sha256'],n
print('MANIFEST VERIFIED',len(m['files']),'files')
