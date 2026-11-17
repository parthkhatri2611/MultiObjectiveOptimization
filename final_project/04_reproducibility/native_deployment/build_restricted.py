"""Create an isolated restricted experiment; never modify the full release."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
FULL=ROOT/'ourwork/campaign_v2_deployment'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    full_hash=sha(FULL/'syftr_campaign_v2.zip')
    with tempfile.TemporaryDirectory(prefix='syftr-restricted-build-') as temp:
        temp=Path(temp); source_root=temp/'project'; frozen=temp/'frozen'
        with zipfile.ZipFile(FULL/'syftr_campaign_v2.zip') as z:
            with zipfile.ZipFile(io.BytesIO(z.read('frozen_common.zip'))) as core:
                core.extractall(frozen)
            for name in z.namelist():
                if name.startswith('source/'):
                    p=source_root/Path(name).relative_to('source')
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
        source_file=frozen/'ourwork/baseline/spec.py'
        original=source_file.read_text()
        old="""    if p['stage'] in {'baseline', 'capabilities', 'pilot'} and p['rag_modes'] != PAPER_MODES:
        raise ValueError('Research profiles must preserve the five paper workflow families')"""
        new="""    if p['stage'] in {'baseline', 'capabilities', 'pilot'} and p['rag_modes'] != PAPER_MODES:
        if p.get('space_variant') != 'restricted-v2' or p['rag_modes'] != ['rag', 'sub_question_rag']:
            raise ValueError('Only the explicitly declared restricted-v2 workflow subset is permitted')"""
        assert original.count(old)==1
        changed=original.replace(old,new)
        changed+='''

# Explicit source variant for the independent restricted-space experiment.
# The original full-release file remains separately archived and unchanged.
_full_search_spec = search_spec
_full_capability_overrides = capability_overrides


def search_spec(protocol):
    d = _full_search_spec(protocol)
    if protocol.get('space_variant') == 'restricted-v2':
        if protocol['rag_modes'] != ['rag', 'sub_question_rag'] or protocol['embedding_models'] != [
                'BAAI/bge-small-en-v1.5', 'BAAI/bge-base-en-v1.5']:
            raise ValueError('Restricted catalog differs from its declared subset')
        d['few_shot_retriever']['top_k']['kmax'] = 4
        d['rag_retriever']['top_k']['kmax'] = 6
        d['rag_retriever']['query_decomposition']['num_queries_max'] = 3
        d['splitter'].update(chunk_max_exp=10, chunk_overlap_frac_max=.2)
        d['reranker']['top_k']['kmax'] = 4
        d['additional_context']['num_nodes_max'] = 4
    return d


def capability_overrides():
    return [(name, params) for name, params in _full_capability_overrides()
            if params['rag_mode'] in {'rag', 'sub_question_rag'}]
'''
        source_file.write_text(changed)
        manifest_file=frozen/'BUNDLE_MANIFEST.json'
        frozen_manifest=json.loads(manifest_file.read_text())
        before=frozen_manifest['ourwork/baseline/spec.py']
        after=sha(source_file)
        frozen_manifest['ourwork/baseline/spec.py']=after
        manifest_file.write_text(json.dumps(frozen_manifest,indent=2)+'\n')
        protocol_file=source_root/'ourwork/campaign_v2/protocol.json'
        protocol=json.loads(protocol_file.read_text())
        protocol.update(campaign_version='restricted-v2-25x10',space_variant='restricted-v2',
            rag_modes=['rag','sub_question_rag'],embedding_models=['BAAI/bge-small-en-v1.5','BAAI/bge-base-en-v1.5'],
            scope='Explicit restricted Syftr subset; separate experiment from full-space campaign')
        protocol_file.write_text(json.dumps(protocol,indent=2)+'\n')
        amendment={'variant':'restricted-v2','changed_common_file':'ourwork/baseline/spec.py',
            'original_file_sha256':before,'restricted_file_sha256':after,
            'full_release_sha256':full_hash,
            'rule':'Fresh restricted capabilities required; do not reuse full-space results as subset validation.'}
        (source_root/'ourwork/campaign_v2/restricted_source_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
        # All run and readiness paths use a fresh measured subset capability run.
        controller=source_root/'ourwork/campaign_v2/controller.py'
        controller.write_text(controller.read_text().replace('capabilities-f1-gpu1-seed11','capabilities-restricted-v2-seed11'))
        report=source_root/'ourwork/campaign_v2/report.py'
        report.write_text(report.read_text().replace('full Syftr conditional space','restricted Syftr subset')
            .replace('Local Syftr adaptation: search set, seed 11','Restricted Syftr subset: search set, seed 11'))
        shutil.copy2(HERE/'restricted_capabilities.py',source_root/'ourwork/campaign_v2/restricted_capabilities.py')
        shutil.copy2(HERE/'test_restricted.py',source_root/'ourwork/campaign_v2/test_restricted.py')
        # Make the shared integration test exercise the actual restricted catalog.
        test=source_root/'ourwork/campaign_v2/test_campaign.py'
        text=test.read_text()
        needle="            profile_file = directory/'profile.json'"
        injection="""            profile.update(space_variant='restricted-v2',rag_modes=['rag','sub_question_rag'],
                embedding_models=['BAAI/bge-small-en-v1.5','BAAI/bge-base-en-v1.5'])
            from ourwork.baseline.spec import search_spec
            config.search_space=helper.Native(**search_spec(profile))
            cap_file=directory/'capabilities/protocol.json'
            cp=read(cap_file)
            cp['experiment']=dict(profile,stage='capabilities')
            atomic(cap_file,cp)
"""
        assert needle in text
        test.write_text(text.replace(needle,injection+needle))
        installer=(FULL/'install_and_restore.py').read_text()
        installer=installer.replace("'ourwork.campaign_v2.test_campaign', '-q'", "'ourwork.campaign_v2.test_campaign', 'ourwork.campaign_v2.test_restricted', '-q'")
        installer=installer.replace("'checked_source_identity(source_identity(Path.cwd()), cap[\"source_identity\"]); '",
            "'from ourwork.baseline.records import fingerprint; import hashlib; '\n        'expected=json.loads(json.dumps(cap[\"source_identity\"])); '\n        'amendment=json.loads(Path(\"ourwork/campaign_v2/restricted_source_amendment.json\").read_text()); '\n        'assert hashlib.sha256(Path(\"ourwork/baseline/spec.py\").read_bytes()).hexdigest()==amendment[\"restricted_file_sha256\"]; '\n        'expected[\"files\"][\"ourwork/baseline/spec.py\"]=amendment[\"restricted_file_sha256\"]; '\n        'expected[\"sha256\"]=fingerprint(expected[\"files\"]); '\n        'checked_source_identity(source_identity(Path.cwd()), expected); '")
        needle="    from ourwork.campaign_v2.snapshots import snapshot as save_snapshot"
        new_step="""    status('fresh_restricted_capabilities')
    subprocess.run([str(python), '-u', '-m', 'ourwork.campaign_v2.restricted_capabilities'], cwd=ROOT, check=True)
"""
        assert needle in installer
        installer=installer.replace(needle,new_step+needle)
        installer=installer.replace("'expected_test_count':53", "'expected_test_count':54")
        installer=installer.replace("'deadline-v2-25x10'", "'restricted-v2-25x10'")
        installer=installer.replace("'New campaign after loss of old unexported baseline. Original 30x20 campaign remains historical.'", "'Independent restricted-space backup study; do not pool with full-space results.'")
        # Capabilities helper/readiness take source revision pins from archived full records,
        # but the actual search arms require the new restricted capability run above.
        (HERE/'install_and_restore.py').write_text(installer)
        cache_target=source_root/'ourwork/kaggle_returns/gpu1-recovery'
        cache_target.mkdir(parents=True)
        (cache_target/'gpu1_checkpoint.tar.gz').symlink_to(ROOT/'ourwork/kaggle_returns/gpu1-recovery/gpu1_checkpoint.tar.gz')
        spec=importlib.util.spec_from_file_location('full_builder',FULL/'build_bundle.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        builder.ROOT=source_root;builder.HERE=HERE;builder.FROZEN=frozen
        builder.main()
    # Give the friend-facing artifacts distinctive filenames and input lookup names.
    (HERE/'syftr_campaign_v2.zip').rename(HERE/'syftr_restricted_v2.zip')
    for old,new in [('SYFTR_FOUR_METHODS_V2.ipynb','SYFTR_RESTRICTED_V2.ipynb'),
                    ('SYFTR_FOUR_METHODS_V2_BATCH.ipynb','SYFTR_RESTRICTED_V2_BATCH.ipynb'),
                    ('KAGGLE_CELL_1.py','KAGGLE_RESTRICTED_CELL_1.py'),
                    ('KAGGLE_BATCH_ALL.py','KAGGLE_RESTRICTED_BATCH_ALL.py')]:
        p=HERE/old
        text=p.read_text().replace('syftr_campaign_v2.zip','syftr_restricted_v2.zip')
        text=text.replace('Four-method campaign: committed batch run','Restricted Syftr subset: four-method batch run')
        text=text.replace('Full search space.','Restricted RAG/SubQuestion space.')
        (HERE/new).write_text(text);p.unlink()
    verification=json.loads((HERE/'PACKAGE_VERIFICATION.json').read_text())
    verification['archive']=str(HERE/'syftr_restricted_v2.zip')
    verification['variant']='restricted-v2'
    (HERE/'PACKAGE_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
    assert sha(FULL/'syftr_campaign_v2.zip')==full_hash,'Full-space release was changed'
    print('RESTRICTED PACKAGE READY; full-space package unchanged.')


if __name__=='__main__':main()
