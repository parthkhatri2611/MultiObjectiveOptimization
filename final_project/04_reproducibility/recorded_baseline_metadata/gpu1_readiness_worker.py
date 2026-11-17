from pathlib import Path
import json,math,time
from ourwork.baseline.adapter import prepare
from ourwork.baseline.device_policy import device_diagnostics
from ourwork.baseline.records import source_identity,atomic_json
root=Path('/kaggle/working/syftr_project')
base=Path('/kaggle/working/baseline_results')
p=json.loads((base/'selected_protocols/pilot_f1_gpu1_v1.json').read_text())
sources=json.loads((base/'capabilities-seed11/sources.json').read_text())
study,builder,hf=prepare(p,'/kaggle/temp/rm-syftr/evaluation',sources)
import torch
embedding=hf.get_hf_embedding_model('BAAI/bge-large-en-v1.5',timeout_config=study.timeouts,device='cpu')
assert str(embedding._model.device) in {'cuda','cuda:0'}
texts=[('This is an embedding throughput readiness test. '*90)+str(i) for i in range(32)]
torch.cuda.synchronize();start=time.monotonic()
vectors=embedding.get_text_embedding_batch(texts)
torch.cuda.synchronize();elapsed=time.monotonic()-start
assert len(vectors)==32 and all(len(v)==1024 and all(math.isfinite(x) for x in v) for v in vectors)
record={'status':'passed','source_identity':source_identity(root),'device':device_diagnostics(p),
        'actual_model_device':str(embedding._model.device),'batch_texts':32,'dimensions':1024,
        'batch_compute_seconds':elapsed,'note':'Synthetic readiness batch, not a benchmark against the earlier full CPU index.'}
atomic_json(base/'gpu1_readiness.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='source_identity'},indent=2))
