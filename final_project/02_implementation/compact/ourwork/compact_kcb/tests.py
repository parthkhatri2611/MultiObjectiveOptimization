import copy,json,tempfile,unittest,warnings
from pathlib import Path
from .measure import clip_context,chunks_for,measure_config
from .replay import run_replays,grid_points,METHODS
from .audit import audit_table
from ourwork.baseline.records import fingerprint

class Tokenizer:
    def encode(self,text,**kwargs):return list(text.encode())
    def decode(self,ids,**kwargs):return bytes(ids).decode(errors='ignore')

class CompactTests(unittest.TestCase):
    def test_unicode_context_hard_budget(self):
        t=Tokenizer()
        for budget in range(1,20):
            text,count=clip_context('abcdef 🐱 café '*10,budget,t)
            self.assertEqual(len(t.encode(text)),count);self.assertLessEqual(count,budget)
        chunks=chunks_for(['abcdefghijk'],4,.25,t)
        self.assertEqual([r['start'] for r in chunks],[0,3,6,9])

    def test_progressive_resume_and_raw_audit(self):
        class Retrieval:
            tokenizer=Tokenizer()
            def retrieve(self,*a):return 'abc',{'context_tokens':3,'requested_context_budget':512}
        class Generator:
            calls=0
            def answer(self,part,qa,context):
                self.calls+=1
                return {'answer':'Paris','error':None,'usage':{'prompt_tokens':12,'completion_tokens':2},'call_id':qa['id'],'key':qa['id'],'cache_hit':False,'elapsed_seconds':.1}
        g=Generator();p={'k':2,'c':256,'Bc':512};rows=[{'id':str(i),'question':'City?','reference':'Paris'} for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            measure_config(p,'test',rows[:1],Retrieval(),g,d)
            result=measure_config(p,'test',rows,Retrieval(),g,d)
            self.assertEqual(g.calls,3);self.assertEqual(result['values'],[1.,14.])
            table={fingerprint(p):result};self.assertTrue(audit_table(table,Tokenizer()))
            broken=copy.deepcopy(table);broken[fingerprint(p)]['values'][0]=.5
            with self.assertRaises(AssertionError):audit_table(broken,Tokenizer())

    def test_all_optimizers_fixed_table_and_charged_queries(self):
        protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());protocol['optimizer_seeds']=[11,23]
        table={fingerprint(p):{'params':p,'values':[i/27,300+i*70],'known_generator_tokens':300+i*70} for i,p in enumerate(grid_points(protocol))}
        with tempfile.TemporaryDirectory() as d,warnings.catch_warnings():
            warnings.simplefilter('ignore');r=run_replays(table,protocol,d)
            self.assertEqual(len(r['results']),8);self.assertTrue(r['global_grid_front_identified'])
            for seed in protocol['optimizer_seeds']:
                runs=[v for v in r['results'] if v['seed']==seed]
                self.assertEqual(len({v['shared_initialization_sha256'] for v in runs}),1)
                self.assertEqual(len({json.dumps([t['params'] for t in v['trace'][:10]]) for v in runs}),1)
                for v in runs:
                    self.assertEqual(len(v['trace']),25)
                    self.assertGreaterEqual(v['hv_gap_to_measured_grid'],-1e-12)
                    self.assertEqual(v['repeat_queries'],25-v['unique_configurations'])
                    self.assertTrue(all(t['values']==table[t['configuration_id']]['values'] for t in v['trace']))
                    if v['method']!='syftr_style_motpe':self.assertEqual(len(v['generation_log']),3)
                    if v['method']=='hybrid_motpe':self.assertEqual(sum(t['proposal_origin']=='motpe_injection' for t in v['trace']),6)
            from .report import report
            report(table,r,protocol,Path(d)/'report',5)
            self.assertTrue((Path(d)/'report/pareto.png').is_file())
