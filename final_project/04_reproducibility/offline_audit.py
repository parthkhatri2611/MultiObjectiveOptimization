"""Audit archived native measurements without GPU, network, or sampler unpickling."""
from pathlib import Path
import sys,json,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'02_implementation/native_exact'))
from ourwork.campaign_v2.report import compare
source=ROOT/'03_results/native_restricted/campaign_artifacts/campaign-v2'
with tempfile.TemporaryDirectory(prefix='syftr-final-audit-') as folder:
 target=Path(folder)/'campaign-v2';shutil.copytree(source,target)
 # The saved selection contains original Kaggle source_run paths. Only remove this
 # derived file in the disposable COPY; do not alter original selections or results.
 (target/'comparison/confirmation_selection.json').unlink()
 rows=compare(target,make_plots=False,prefix=25)
 saved=json.loads((ROOT/'03_results/native_restricted/AUDIT_at_25.json').read_text())
 for actual,expected in zip(rows,saved['methods']):
  for k in ['method','attempts','completed','failed','hypervolume','best_f1','known_generator_tokens','known_judge_tokens']:
   assert actual[k]==expected[k],(k,actual[k],expected[k])
 output={'status':'passed','eligible_candidates':sum(r['completed'] for r in rows),'qa_rows':10*sum(r['completed'] for r in rows),'methods':rows,'holdout':'No usable native holdout result'}
 (ROOT/'04_reproducibility/OFFLINE_AUDIT_RESULT.json').write_text(json.dumps(output,indent=2)+'\n')
 for r in rows: print(r['method'],'HV',r['hypervolume'],'best F1',r['best_f1'])
 print('RAW OBJECTIVE AUDIT PASSED; original archived campaign was not modified.')
