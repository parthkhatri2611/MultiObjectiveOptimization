from pathlib import Path
import hashlib,json
from huggingface_hub import hf_hub_download
base=Path('/kaggle/working/baseline_results')
meta=json.loads((base/'server_metadata.json').read_text())
paths={}
for name,expected in meta['files_sha256'].items():
    p=Path(hf_hub_download(meta['model_repo'],filename=name,revision=meta['model_revision'],cache_dir='/kaggle/temp/rm-syftr/server/weights'))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==expected,name
    paths[name]=str(p)
    print('WEIGHT VERIFIED',name,flush=True)
original_model=meta['command'][meta['command'].index('-m')+1]
assert paths[Path(original_model).name]==original_model,'Original weight path was not restored'
