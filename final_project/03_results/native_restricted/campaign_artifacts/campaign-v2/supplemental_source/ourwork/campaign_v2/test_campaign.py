"""Software fixtures only: synthetic answers verify plumbing, never research findings."""
import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings
import zipfile
import optuna
from ourwork.campaign_v2.common import atomic, read, workspace_for, implementation_identity, METHODS
from ourwork.campaign_v2.run import run_arm
from ourwork.campaign_v2.report import compare, comparable, coverage, validate_protocols
from ourwork.campaign_v2.snapshots import snapshot
from ourwork.campaign_v2.confirm import paired_bootstrap
from ourwork.optimizers import test_integration as fixture_module
from ourwork.baseline.records import append_jsonl
from ourwork.baseline.metrics import answer_scores
from ourwork.baseline.worker import summarize


class CampaignUtilitiesTests(unittest.TestCase):
    def test_shared_cache_private_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = workspace_for('syftr_motpe', 11, tmp)
            b = workspace_for('nsga2', 11, tmp)
            self.assertNotEqual(a, b)
            self.assertEqual((a/'indexes').resolve(), (b/'indexes').resolve())
            self.assertEqual(workspace_for('nsga2', 11, tmp), b)
            (a/'private-results').write_text('private')
            self.assertFalse((b/'private-results').exists())

    def test_sqlite_wal_capsule_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)/'campaign'
            campaign.mkdir()
            db = sqlite3.connect(campaign/'study.sqlite3')
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE scores (value INTEGER)')
            db.execute('INSERT INTO scores VALUES (17)')
            db.commit()
            atomic(campaign/'state.json', {'measured': False})
            (campaign/'.controller.lock').touch()
            receipt = snapshot(campaign, Path(tmp)/'exports')
            with zipfile.ZipFile(Path(tmp)/'exports'/receipt['file']) as z:
                manifest = json.loads(z.read('CAPSULE_MANIFEST.json'))
                for name, item in manifest.items():
                    data = z.read(name)
                    self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])
                self.assertFalse(any(n.endswith(('-wal','-shm','.lock')) for n in z.namelist()))
                recovered = Path(tmp)/'recovered.sqlite3'
                recovered.write_bytes(z.read('campaign-v2/study.sqlite3'))
            with sqlite3.connect(recovered) as restored:
                self.assertEqual(restored.execute('SELECT value FROM scores').fetchone(), (17,))
            db.close()

    def test_zip_and_expanded_capsule_recovery(self):
        from ourwork.campaign_v2.recovery import restore_capsule
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=root/'original'
            atomic(original/'runs/example/state.json', {'attempts':7,'fixture_only':True})
            snapshot(original,root/'exports')
            restore_capsule(root/'output/campaign-v2',root/'exports',root/'output')
            self.assertEqual(read(root/'output/campaign-v2/runs/example/state.json')['attempts'],7)
            # Do not overwrite newer in-session trials with an older input capsule.
            atomic(root/'output/campaign-v2/runs/example/state.json',{'attempts':8})
            restore_capsule(root/'output/campaign-v2',root/'exports',root/'output')
            self.assertEqual(read(root/'output/campaign-v2/runs/example/state.json')['attempts'],8)
            expanded=root/'expanded'
            with zipfile.ZipFile(root/'exports/campaign_v2_review.zip') as z:
                z.extractall(expanded)
            restore_capsule(root/'output2/campaign-v2',expanded,root/'output2')
            self.assertEqual(read(root/'output2/campaign-v2/runs/example/state.json')['attempts'],7)
            (expanded/'campaign-v2/runs/example/state.json').write_text('tampered')
            with self.assertRaises(AssertionError):
                restore_capsule(root/'output3/campaign-v2',expanded,root/'output3')

    def test_only_nonrepository_git_diagnostic_can_differ(self):
        from ourwork.campaign_v2.common import checked_source_identity
        expected={'sha256':'same','files':{'code.py':'hash'},'git_head':{
            'returncode':128,'stdout':'','stderr':'fatal: not a git repository (mount boundary)'}}
        actual=json.loads(json.dumps(expected))
        actual['git_head']['stderr']='fatal: not a git repository (parent directory)'
        self.assertEqual(checked_source_identity(actual,expected),expected)
        actual['files']['code.py']='modified'
        with self.assertRaises(ValueError):
            checked_source_identity(actual,expected)
        actual=json.loads(json.dumps(expected))
        actual['git_head']={'returncode':0,'stdout':'another revision','stderr':''}
        with self.assertRaises(ValueError):
            checked_source_identity(actual,expected)

    def test_capsule_rejects_parent_paths(self):
        from ourwork.campaign_v2.recovery import check_paths
        with self.assertRaises(AssertionError):
            check_paths(['campaign-v2/../../escape'])
        with self.assertRaises(AssertionError):
            check_paths(['/absolute/path'])

    def test_paired_bootstrap_zero_and_mismatch(self):
        values = {'q1': .5, 'q2': 1}
        result = paired_bootstrap(values, values)
        self.assertEqual(result['question_bootstrap_95_percentile_interval'], [0, 0])
        with self.assertRaises(ValueError):
            paired_bootstrap(values, {'q1': .5})

    def test_coverage_and_allowed_protocol_metadata(self):
        self.assertEqual(coverage([(1, 1)], [(.5, 2)]), 1)
        self.assertEqual(coverage([], [(.5, 2)]), 0)
        self.assertIsNone(coverage([], []))
        a = {'experiment': {'budget': 25, 'optimizer_method': 'a', 'population_size': 5}}
        b = {'experiment': {'budget': 25, 'optimizer_method': 'b'}}
        self.assertEqual(comparable(a), comparable(b))
        b['experiment']['budget'] = 24
        self.assertNotEqual(comparable(a), comparable(b))


class BackgroundControlTests(unittest.TestCase):
    def test_notebook_reads_controller_status_path(self):
        from ourwork.campaign_v2 import notebook
        with tempfile.TemporaryDirectory() as tmp:
            atomic(Path(tmp)/'status.json', {'status':'paused','active_method':'nsga2'})
            with patch.object(notebook,'CAMPAIGN',Path(tmp)), redirect_stdout(io.StringIO()):
                result=notebook.status()
            self.assertEqual(result['status'],'paused')
            self.assertEqual(result['active_method'],'nsga2')

    def test_background_driver_serial_dispatch_and_soft_stop(self):
        from ourwork.campaign_v2 import controller
        with tempfile.TemporaryDirectory() as tmp:
            campaign=Path(tmp)
            atomic(campaign/'protocol.json', {'budget':25})
            atomic(campaign/'plotting_python.json', {'executable':sys.executable})
            dispatched=[]
            class Process:
                pid=12345
                def wait(self,timeout):
                    return 0
            def spawn(command,**kwargs):
                method=command[command.index('--method')+1]
                dispatched.append(method)
                output=Path(command[command.index('--output')+1])
                target=int(command[command.index('--target-attempts')+1])
                atomic(output/'analysis/summary.json', {'budget_exhausted':target==25,'attempts':target})
                return Process()
            with patch.object(controller.subprocess,'Popen',side_effect=spawn), \
                 patch.object(controller.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)), \
                 patch.object(controller,'snapshot'), \
                 patch.object(controller,'workspace_for',return_value=campaign/'workspace'), \
                 patch('ourwork.campaign_v2.report.compare') as report:
                controller.control(campaign,hours=1)
                self.assertEqual(dispatched,METHODS*4)
                self.assertEqual(read(campaign/'status.json')['status'],'search_complete')
                self.assertEqual(report.call_count,4)
                report.assert_called_with(campaign,make_plots=False,prefix=25)
                (campaign/'STOP_REQUESTED').touch()
                controller.control(campaign,hours=1)
                self.assertEqual(read(campaign/'status.json')['status'],'paused')
                self.assertEqual(dispatched,METHODS*4)


class CampaignIntegrationTests(unittest.TestCase):
    def test_four_actual_runner_arms_pause_resume_and_raw_audit(self):
        # Frozen Syftr's native sampling code, real Optuna/SQLite and shared run loop.
        # Only GPU/model environment and worker inference are synthetic fixtures.
        fixture_module.SharedRunnerTests.setUpClass()
        helper = fixture_module.SharedRunnerTests()
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            directory = Path(tmp)
            protocol, sources, identity, config, environment = helper.fixture(directory)
            profile = dict(protocol, budget=25, qa_count=10, campaign_version='deadline-v2-25x10',
                           cache_policy='shared-pinned-retrieval-indexes-v2')
            profile.update(space_variant='restricted-v2',rag_modes=['rag','sub_question_rag'],
                embedding_models=['BAAI/bge-small-en-v1.5','BAAI/bge-base-en-v1.5'])
            from ourwork.baseline.spec import search_spec
            config.search_space=helper.Native(**search_spec(profile))
            cap_file=directory/'capabilities/protocol.json'
            cp=read(cap_file)
            cp['experiment']=dict(profile,stage='capabilities')
            atomic(cap_file,cp)
            profile_file = directory/'profile.json'
            atomic(profile_file, profile)
            campaign = directory/'campaign'
            campaign.mkdir()
            atomic(campaign/'implementation.json', implementation_identity())
            manifest = {'selected_ids': [f'q{i}' for i in range(10)], 'questions_sha256':'fixture',
                        'dataset':'hotpotqa','partition':'test'}
            fake_syftr = types.ModuleType('syftr')
            fake_syftr.__path__ = []
            fake_studies = types.ModuleType('syftr.studies')
            fake_studies.SearchSpace = helper.Native
            pause_once = [True]
            def worker(job, timeout, root):
                dest = Path(job['destination'])
                dest.mkdir(parents=True,exist_ok=True)
                atomic(dest/'job.json',job)
                number = int(dest.name.split('-')[0])
                salt = int(job['candidate_id'][:8],16)
                rows=[]
                for i,qid in enumerate(manifest['selected_ids']):
                    answer = ['alpha beta','alpha','gamma'][(salt+i)%3]
                    usage={'prompt_tokens':100+salt%800,'completion_tokens':5}
                    call={'call_id':f'fixture-{number}-{i}','role':'generation','qa_id':qid,
                          'usage':usage,'usage_verified':True,'http_status':200,'elapsed_seconds':.01}
                    append_jsonl(dest/'calls.jsonl',call)
                    append_jsonl(dest/'requests.jsonl',{'call_id':call['call_id'],'role':'generation'})
                    row={'qa_id':qid,'candidate_id':job['candidate_id'],'question':'fixture question',
                         'reference':'alpha beta','answer':answer,'generation_error':None,'judge_error':None,
                         'score':4,'passing':True,'feedback':'synthetic','truncated':False,
                         'generation_seconds':.01,'judge_seconds':0,'qtype':'synthetic','difficulty':'synthetic',
                         'calls':[call],**answer_scores(answer,'alpha beta')}
                    append_jsonl(dest/'questions.jsonl',row)
                    rows.append(row)
                metrics=summarize(rows,10)
                metrics.update(construction_seconds=.01,candidate_seconds=.1,construction_model_calls=0)
                result={'status':'evaluated','metrics':metrics}
                atomic(dest/'result.json',result)
                if pause_once[0] and number==2:
                    pause_once[0]=False
                    (campaign/'STOP_REQUESTED').touch()
                return result
            patches=[
                patch.dict(sys.modules,{'syftr':fake_syftr,'syftr.studies':fake_studies,
                                       'actual_syftr_space_test':helper.native_module}),
                patch('ourwork.baseline.adapter.configure'),
                patch('ourwork.baseline.run.source_identity',return_value=identity),
                patch('ourwork.baseline.run.environment',return_value=environment),
                patch('ourwork.baseline.run.prepare',return_value=(config,None,None)),
                patch('ourwork.baseline.run.question_manifest',return_value=([],manifest)),
                patch('ourwork.baseline.run.grounding_manifest',return_value={'fixture':True}),
                patch('ourwork.baseline.run.execute_job',side_effect=worker),
                patch('ourwork.campaign_v2.run.snapshot'),
            ]
            with ExitStack() as stack:
                for item in patches:
                    stack.enter_context(item)
                stack.enter_context(redirect_stdout(io.StringIO()))
                optuna.logging.set_verbosity(optuna.logging.ERROR)
                for target in [10,15,20,25]:
                    for method in METHODS:
                        output=campaign/'runs'/f'{method}-seed11'
                        args=argparse.Namespace(method=method,protocol=str(profile_file),campaign=str(campaign),
                            output=str(output),workspace=str(directory/f'work-{method}'),seed=11,max_hours=4,
                            capabilities_run=str(directory/'capabilities'),server_metadata=str(directory/'server.json'),
                            preflight=str(directory/'preflight.json'),export_dir=str(directory/'exports'),
                            baseline_run=str(campaign/'runs/syftr_motpe-seed11'),target_attempts=target)
                        if method=='syftr_motpe' and target==10:
                            self.assertFalse(run_arm(args))
                            self.assertEqual(read(output/'analysis/summary.json')['attempts'],3)
                            (campaign/'STOP_REQUESTED').unlink()
                        self.assertEqual(run_arm(args),target==25)
                        study=optuna.load_study(study_name='baseline',storage=f'sqlite:///{output/"study.sqlite3"}')
                        self.assertEqual(len(study.trials),target)
                        if method!='syftr_motpe' and target==25:
                            self.assertEqual(len(study.user_attrs['optimizer_generations']),3)
                            injections=sum(t.user_attrs.get('proposal_origin')=='motpe_injection' for t in study.trials)
                            self.assertEqual(injections,6 if method=='hybrid_motpe' else 0)
                    interim=compare(campaign,make_plots=False,prefix=target)
                    self.assertTrue(all(r['attempts']==target for r in interim))
                    if target<25:
                        self.assertFalse((campaign/f'comparison_at_{target}/confirmation_selection.json').exists())
                plots_available = importlib.util.find_spec("matplotlib") is not None
                rows=compare(campaign,make_plots=plots_available)
                self.assertEqual(len(rows),4)
                for filename in ['pareto.png','pareto.pdf','convergence_attempts.png','convergence_tokens.png']:
                    if plots_available:
                        self.assertGreater((campaign/'comparison'/filename).stat().st_size,100)
                self.assertEqual(len(read(campaign/'comparison/confirmation_selection.json')),4)
                # Different question counts must be rejected before making comparative claims.
                altered=campaign/'runs/nsga2-seed11/protocol.json'
                old=read(altered)
                changed=json.loads(json.dumps(old))
                changed['experiment']['qa_count']=9
                atomic(altered,changed)
                with self.assertRaisesRegex(ValueError,'protocols'):
                    compare(campaign,make_plots=False)
                atomic(altered,old)
                # A corrupted answer must fail the raw-objective audit.
                qfile=next((campaign/'runs/nsga2-seed11/candidates').glob('*/questions.jsonl'))
                original=qfile.read_text()
                records=[json.loads(line) for line in original.splitlines()]
                records[0]['answer']='unrecorded changed answer'
                qfile.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
                with self.assertRaisesRegex(ValueError,'F1/EM'):
                    compare(campaign,make_plots=False)
                qfile.write_text(original)


if __name__=='__main__':
    unittest.main()
