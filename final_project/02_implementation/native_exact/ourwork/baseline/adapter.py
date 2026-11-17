"""Reuse Syftr's real SearchSpace and flow builder, with explicit local adaptations."""
import importlib
import json
import os
from pathlib import Path
import random
from .records import atomic_json, fingerprint
from .spec import DATASETS, search_spec
from .device_policy import configure_embedding_device

JUDGE_SYSTEM = '''You evaluate factual correctness against the supplied reference answer.
Do not use outside knowledge. Score 1 if irrelevant, 2 or 3 if relevant but incorrect,
and 4 or 5 if correct and consistent with the reference. Return only a JSON object:
{"score": 4, "reasoning": "The answer agrees with the reference."}'''


def configure(protocol, workspace, source_manifest):
    """Must execute before importing any syftr module."""
    configure_embedding_device(protocol)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    paths = {key: str(workspace / key) for key in
             ['results_dir', 'tmp_dir', 'huggingface_cache', 'onnx_dir', 'lock_dir', 'nltk_dir', 'sqlite_dir']}
    cache_scope = fingerprint({'sources': source_manifest, 'tokenizer_policy': 'generator-tokenizer-v1', 'embedding_device': protocol.get('embedding_device', 'cpu')})[:20]
    paths['index_cache'] = str(workspace / 'indexes' / cache_scope)
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)
    config = {'paths': paths, 'storage': {'s3_cache_enabled': False, 'local_cache_max_size_gb': 8},
              'instrumentation': {'tracing_enabled': False},
              'ray': {'local': True}, 'logging': {'level': 30},
              'generative_models': {protocol['model_alias']: {
                  'provider': 'openai_like', 'model_name': protocol['model_alias'],
                  'api_base': 'http://127.0.0.1:8766/v1', 'api_key': 'local-only',
                  'temperature': 0, 'top_p': 1, 'max_tokens': protocol['max_output_tokens'],
                  'context_window': protocol['context_window'], 'timeout': protocol['request_timeout'],
                  'is_chat_model': True, 'is_function_calling_model': True,
                  'cost': {'type': 'tokens', 'input': 0.0, 'output': 0.0},
              }}}
    path = workspace / 'config.yaml'
    # JSON is valid YAML. No cloud credentials are copied from the machine.
    atomic_json(path, config)
    os.environ['SYFTR_CONFIG_FILE'] = str(path)
    os.environ['SYFTR_PATHS__SQLITE_DIR'] = paths['sqlite_dir']
    os.environ['HF_HOME'] = str(workspace / 'hf-home')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['SYFTR_INSTRUMENTATION__TRACING_ENABLED'] = 'false'
    return config


def resolve_sources(protocol, original):
    from huggingface_hub import HfApi
    api = HfApi()
    sources = json.loads(json.dumps(original))
    models = [protocol['tokenizer_repo']] + protocol['embedding_models']
    sources['embedding_and_tokenizer_revisions'] = {
        name: api.model_info(name).sha for name in models}
    repo = DATASETS[protocol['dataset']]['repo']
    sources['dataset_revisions'] = {repo: sources['dataset']['revision'] if protocol['dataset'] == 'hotpotqa'
                                    else api.dataset_info(repo).sha}
    return sources


def prepare(protocol, workspace, sources):
    configure(protocol, workspace, sources)
    import datasets
    from transformers import AutoTokenizer
    import syftr.llm as llm
    import syftr.storage as storage
    import syftr.huggingface_helper as hf
    from syftr.configuration import cfg
    from syftr.studies import SearchSpace, StudyConfig
    original_load = datasets.load_dataset
    revisions = sources['embedding_and_tokenizer_revisions']

    def pinned_load(path, *args, **kwargs):
        if path not in sources['dataset_revisions']:
            raise ValueError(f'Unpinned dataset access: {path}')
        kwargs['revision'] = sources['dataset_revisions'][path]
        return original_load(path, *args, **kwargs)
    datasets.load_dataset = pinned_load
    original_tokenizer = AutoTokenizer.from_pretrained

    def pinned_tokenizer(name, *args, **kwargs):
        if str(name) in revisions:
            kwargs['revision'] = revisions[str(name)]
        return original_tokenizer(name, *args, **kwargs)
    AutoTokenizer.from_pretrained = pinned_tokenizer
    tokenizer = original_tokenizer(protocol['tokenizer_repo'],
                                    revision=revisions[protocol['tokenizer_repo']],
                                    cache_dir=cfg.paths.huggingface_cache)

    def get_tokenizer(_name):
        def encode(text, **ignored):
            return tokenizer.encode(text, add_special_tokens=False)
        return encode
    llm.get_tokenizer = get_tokenizer

    def pinned_embedding(name, timeout_config=None, total_chunks=0, device='cpu'):
        if name not in revisions:
            raise ValueError(f'Unpinned embedding model: {name}')
        gpu_policy = protocol.get('embedding_device', 'cpu') == 'cuda:1'
        if gpu_policy:
            device = 'cuda:0'  # physical GPU 1 is the only visible GPU in this process
        embedding = hf.HuggingFaceEmbeddingWithTimeout(
            model_name=name, device=device, cache_folder=str(cfg.paths.huggingface_cache),
            revision=revisions[name], trust_remote_code=False, total_chunks=total_chunks,
            timeout_config=timeout_config, embed_batch_size=16)
        if gpu_policy:
            actual_device = str(embedding._model.device)
            if actual_device not in {'cuda', 'cuda:0'}:
                raise RuntimeError(f'Embedding loaded on {actual_device}, expected physical GPU 1')
            print(f'EMBEDDING_READY model={name} physical_gpu=1 logical={actual_device} batch=16', flush=True)
        return embedding
    hf.get_hf_embedding_model = pinned_embedding
    # Imports bind direct function references; patch both current and future users.
    from syftr.tuner import qa_tuner, core
    import syftr.flows as flows
    core.get_tokenizer = flows.get_tokenizer = get_tokenizer
    from llama_index.core import Settings as LlamaSettings
    LlamaSettings.tokenizer = get_tokenizer(protocol['model_alias'])
    # The upstream global token tracker serializes embedding lists incorrectly and
    # is not needed: the proxy journals real endpoint calls/usage independently.
    flows.dispatcher.event_handlers = [h for h in flows.dispatcher.event_handlers if h is not flows._event_handler]
    flows.dispatcher.span_handlers = [h for h in flows.dispatcher.span_handlers if h is not flows._event_handler._span_handler]
    space = SearchSpace(**search_spec(protocol))
    dataset_config = dict(DATASETS[protocol['dataset']])
    dataset_config.pop('repo')
    dataset_config['partition_map'] = {'test': protocol['partition'], 'sample': 'sample',
                                        'train': 'train', 'holdout': 'holdout'}
    study = StudyConfig(name='rm-local-common-evaluator', dataset=dataset_config, search_space=space,
                        recreate_study=False, reuse_study=True, toy_mode=False,
                        evaluation={'mode': 'single', 'llm_names': [protocol['model_alias']],
                                    'eval_system_template': JUDGE_SYSTEM, 'score_threshold': 4.0},
                        optimization={'num_trials': 0, 'use_hf_embedding_models': False,
                                      'embedding_device': ('cuda' if protocol.get('embedding_device', 'cpu') == 'cuda:1' else 'cpu'), 'include_responses': True,
                                      'use_pareto_pruner': False, 'use_cost_pruner': False,
                                      'use_runtime_pruner': False, 'max_concurrent_trials': 1,
                                      'use_individual_baselines': False, 'use_agent_baselines': False,
                                      'use_variations_of_baselines': False})
    return study, qa_tuner, hf


def question_manifest(study, protocol):
    pairs = list(study.dataset.iter_examples())
    ids = [p.id for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate QA identifiers in prepared dataset')
    # Stable selection independent of optimizer seed and original load ordering.
    pairs.sort(key=lambda p: p.id)
    random.Random(protocol['data_seed']).shuffle(pairs)
    if protocol['qa_count'] > len(pairs):
        raise ValueError(f'Requested {protocol["qa_count"]} questions; split has {len(pairs)}')
    pairs = pairs[:protocol['qa_count']]
    return pairs, {'dataset': protocol['dataset'], 'partition': protocol['partition'],
                   'available_questions': len(ids), 'selected_ids': [p.id for p in pairs],
                   'questions_sha256': fingerprint([p.model_dump(mode='json') for p in pairs])}


def grounding_manifest(study):
    import hashlib
    digest, count = hashlib.sha256(), 0
    for doc in study.dataset.iter_grounding_data():
        text = doc.text.encode()
        digest.update(len(text).to_bytes(8, 'big'))
        digest.update(text)
        count += 1
    return {'document_count': count, 'ordered_text_sha256': digest.hexdigest()}
