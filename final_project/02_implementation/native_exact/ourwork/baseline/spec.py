"""One declared Syftr search space for all future optimizer arms."""
from copy import deepcopy
import json
from pathlib import Path

PAPER_MODES = ['rag', 'sub_question_rag', 'critique_rag_agent', 'react_rag_agent', 'lats_rag_agent']
PAPER_EMBEDDINGS = [
    'BAAI/bge-small-en-v1.5', 'BAAI/bge-large-en-v1.5', 'thenlper/gte-large',
    'mixedbread-ai/mxbai-embed-large-v1', 'WhereIsAI/UAE-Large-V1',
    'avsolatorio/GIST-large-Embedding-v0', 'w601sxs/b1ade-embed', 'Labib11/MUG-B-1.6',
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'BAAI/bge-base-en-v1.5',
]
DATASETS = {
    'hotpotqa': {'repo': 'DataRobot-Research/hotpotqa', 'xname': 'hotpotqa_hf', 'subset': 'train_hard'},
    'financebench': {'repo': 'DataRobot-Research/financebench', 'xname': 'financebench_hf'},
    'crag_music': {'repo': 'DataRobot-Research/crag', 'xname': 'crag_hf', 'subset': 'music'},
    'crag_sports': {'repo': 'DataRobot-Research/crag', 'xname': 'crag_hf', 'subset': 'sports'},
    'drdocs': {'repo': 'DataRobot-Research/drdocs', 'xname': 'drdocs_hf'},
    'infinitebench': {'repo': 'xinrongzhang2022/InfiniteBench', 'xname': 'infinitebench_hf', 'subset': 'longbook_qa_eng'},
}


def load_protocol(path):
    p = json.loads(Path(path).read_text())
    if p.get('embedding_device', 'cpu') not in {'cpu', 'cuda:1'}:
        raise ValueError('Unsupported embedding device policy')
    if p['dataset'] not in DATASETS:
        raise ValueError('Unknown dataset')
    if p['partition'] not in {'sample', 'test', 'holdout'}:
        raise ValueError('Unsupported partition')
    if p['partition'] == 'holdout' and p['stage'] != 'confirm':
        raise ValueError('Holdout cannot be searched')
    if p['objective_quality'] not in {'accuracy', 'f1'}:
        raise ValueError('Unsupported quality objective')
    if p['objective_resource'] not in {'total_tokens_mean', 'p80_time'}:
        raise ValueError('Unsupported resource objective')
    if p['budget'] < 1 or p['qa_count'] < 1 or p['startup_trials'] < 1:
        raise ValueError('Budgets and startup counts must be positive')
    if p['stage'] == 'baseline' and p['startup_trials'] >= p['budget']:
        raise ValueError('Baseline budget must leave room for guided TPE')
    if not set(p['embedding_models']).issubset(PAPER_EMBEDDINGS):
        raise ValueError('Embedding catalog is not a subset of paper models')
    if p['stage'] in {'baseline', 'capabilities', 'pilot'} and p['rag_modes'] != PAPER_MODES:
        if p.get('space_variant') != 'restricted-v2' or p['rag_modes'] != ['rag', 'sub_question_rag']:
            raise ValueError('Only the explicitly declared restricted-v2 workflow subset is permitted')
    if not p['embedding_models'] or len(p['embedding_models']) != len(set(p['embedding_models'])):
        raise ValueError('Embedding catalog must be nonempty and unique')
    return p


def llm_spec(alias):
    # Fixed decoding and one available generator replace the paper's hosted LLM catalog.
    return dict(llm_names=[alias], llm_temperature_min=0.0, llm_temperature_max=0.0,
                llm_top_p_min=1.0, llm_top_p_max=1.0, llm_use_reasoning=[None])


def search_spec(protocol):
    llm = llm_spec(protocol['model_alias'])
    d = {
        'rag_modes': protocol['rag_modes'], 'template_names': ['default', 'concise', 'CoT'],
        'response_synthesizer_llm_config': deepcopy(llm),
        'few_shot_enabled': [True, False], 'hyde_enabled': [True, False],
        'reranker_enabled': [True, False], 'additional_context_enabled': [True, False],
        'few_shot_retriever': {'embedding_models': protocol['embedding_models'],
                               'top_k': {'kmin': 2, 'kmax': 20, 'log': False}},
        'splitter': {'methods': ['recursive', 'sentence', 'token'], 'chunk_min_exp': 8,
                     'chunk_max_exp': 12, 'chunk_overlap_frac_min': 0.0,
                     'chunk_overlap_frac_max': 0.7, 'chunk_overlap_frac_step': 0.1},
        'rag_retriever': {'methods': ['dense', 'sparse', 'hybrid'],
                          'embedding_models': protocol['embedding_models'],
                          'top_k': {'kmin': 2, 'kmax': 20, 'log': False},
                          'query_decomposition_enabled': [True, False],
                          'query_decomposition': {'llm_config': deepcopy(llm),
                                                  'num_queries_min': 2, 'num_queries_max': 20,
                                                  'num_queries_step': 1}},
        'reranker': {'llm_config': deepcopy(llm), 'top_k': {'kmin': 2, 'kmax': 20, 'log': False}},
        'hyde': {'llm_config': deepcopy(llm)},
        'additional_context': {'num_nodes_min': 2, 'num_nodes_max': 20},
    }
    for name in ['react_rag_agent', 'sub_question_rag', 'critique_rag_agent']:
        d[name] = {'subquestion_engine_llm_config': deepcopy(llm),
                   'subquestion_response_synthesizer_llm_config': deepcopy(llm)}
    d['critique_rag_agent'].update(critique_agent_llm_config=deepcopy(llm),
                                   reflection_agent_llm_config=deepcopy(llm))
    if protocol['stage'] == 'smoke':
        d.update(few_shot_enabled=[False], hyde_enabled=[False], reranker_enabled=[False],
                 additional_context_enabled=[False])
        d['rag_retriever']['query_decomposition_enabled'] = [False]
    return d


def capability_overrides():
    base = {'rag_mode': 'rag', 'template_name': 'concise', 'rag_method': 'dense',
            'rag_top_k': 2, 'splitter_method': 'sentence', 'splitter_chunk_exp': 9,
            'splitter_chunk_overlap_frac': 0.1, 'few_shot_enabled': False,
            'hyde_enabled': False, 'reranker_enabled': False,
            'additional_context_enabled': False, 'rag_query_decomposition_enabled': False}
    cases = [(mode, {**base, 'rag_mode': mode}) for mode in PAPER_MODES]
    cases += [(name, {**base, **override}) for name, override in [
        ('sparse', {'rag_method': 'sparse'}), ('hybrid', {'rag_method': 'hybrid'}),
        ('few_shot', {'few_shot_enabled': True}), ('HyDE', {'hyde_enabled': True}),
        ('reranker', {'reranker_enabled': True, 'reranker_top_k': 2}),
        ('neighbors', {'additional_context_enabled': True, 'additional_context_num_nodes': 2}),
        ('decomposition', {'rag_query_decomposition_enabled': True, 'rag_query_decomposition_num_queries': 2}),
        ('token_splitter', {'splitter_method': 'token'}),
        ('recursive_splitter', {'splitter_method': 'recursive'}),
    ]]
    return cases


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
