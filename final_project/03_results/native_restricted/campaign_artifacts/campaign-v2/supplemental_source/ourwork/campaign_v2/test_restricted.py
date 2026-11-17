"""Native-space sampling verifies the precise restriction; no model inference."""
import json
import hashlib
from pathlib import Path
import unittest
import optuna
from ourwork.baseline.spec import load_protocol,search_spec,capability_overrides
from ourwork.baseline.test_baseline import actual_space_class


class RestrictedSpaceTest(unittest.TestCase):
    def test_declared_subset_and_native_candidate_bounds(self):
        root=Path(__file__).resolve().parents[2]
        protocol=load_protocol(root/'ourwork/campaign_v2/protocol.json')
        amendment=json.loads((root/'ourwork/campaign_v2/restricted_source_amendment.json').read_text())
        self.assertEqual(hashlib.sha256((root/'ourwork/baseline/spec.py').read_bytes()).hexdigest(),amendment['restricted_file_sha256'])
        self.assertEqual(protocol['space_variant'],'restricted-v2')
        self.assertEqual(protocol['rag_modes'],['rag','sub_question_rag'])
        self.assertEqual(len(protocol['embedding_models']),2)
        self.assertEqual(len(capability_overrides()),11)
        space=actual_space_class()(**search_spec(protocol))
        study=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11),directions=['maximize','minimize'])
        seen=set()
        for _ in range(200):
            trial=study.ask();p=space.sample(trial);seen.add(p['rag_mode'])
            self.assertIn(p['rag_mode'],protocol['rag_modes'])
            for key in ['rag_embedding_model','few_shot_embedding_model']:
                if key in p:self.assertIn(p[key],protocol['embedding_models'])
            for key,bound in [('rag_top_k',6),('few_shot_top_k',4),('rag_query_decomposition_num_queries',3),
                              ('reranker_top_k',4),('additional_context_num_nodes',4),('splitter_chunk_exp',10)]:
                if key in p:self.assertLessEqual(p[key],bound)
            self.assertLessEqual(p['splitter_chunk_overlap_frac'],.2)
            study.tell(trial,(.5,1000))
        self.assertEqual(seen,set(protocol['rag_modes']))
