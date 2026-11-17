"""Measure all representative cases in the declared subset before its search."""
import argparse
from pathlib import Path
import shutil
from unittest.mock import patch
from .common import CAMPAIGN, OUTPUT, ROOT, atomic, read, workspace_for, checked_source_identity


def main():
    import ourwork.baseline.run as common
    from ourwork.baseline.spec import capability_overrides, load_protocol, search_spec
    from ourwork.campaign_v2.snapshots import snapshot
    profile=load_protocol(ROOT/'ourwork/campaign_v2/protocol.json')
    pinned=read(OUTPUT/'capabilities-f1-gpu1-seed11/protocol.json')['sources']
    output=OUTPUT/'capabilities-restricted-v2-seed11'
    archived=CAMPAIGN/'restricted_capabilities'
    if not output.exists() and archived.exists():
        shutil.copytree(archived,output)
    expected=dict(profile,stage='capabilities',budget=len(capability_overrides()),qa_count=3,startup_trials=5)
    if (output/'protocol.json').exists():
        prior=read(output/'protocol.json')
        assert prior['experiment']==expected and prior['sources']==pinned
        checked_source_identity(common.source_identity(ROOT),prior['source_identity'])
        if read(output/'analysis/summary.json')['completed']==len(capability_overrides()):
            print('PREVIOUS_RESTRICTED_CAPABILITIES_VERIFIED',flush=True)
            return
    selected=CAMPAIGN/'restricted_capability_protocol.json'
    atomic(selected,expected)
    args=argparse.Namespace(protocol=str(selected),output=str(output),
        workspace=str(workspace_for('restricted-capabilities',11)),seed=11,max_hours=1.5,
        resume=(output/'protocol.json').exists(),capabilities_run=None,
        server_metadata=str(OUTPUT/'server_metadata.json'),preflight=str(OUTPUT/'preflight_gpu1_session.json'))
    with patch.object(common,'resolve_sources',return_value=pinned):
        common.run(args)
    summary=read(output/'analysis/summary.json')
    if summary['completed']!=len(capability_overrides()) or summary['failed']:
        raise RuntimeError('Restricted capabilities incomplete or failed. Return the log; do not start research arms.')
    checks=read(output/'embedding_checks.json')
    assert all(checks.get(name,{}).get('status')=='passed' for name in profile['embedding_models'])
    shutil.copytree(output,archived,dirs_exist_ok=True)
    snapshot(CAMPAIGN)
    print('RESTRICTED_CAPABILITIES_PASSED',summary,flush=True)


if __name__=='__main__':main()
