"""One candidate in an isolated process; shared evaluator for future optimizers."""
import argparse
import asyncio
import gc
import json
from pathlib import Path
import random
import time
import traceback
from .adapter import prepare, question_manifest
from .metrics import answer_scores, percentile
from .proxy import CallProxy
from .records import append_jsonl, atomic_json, fingerprint, read_jsonl


def summarize(rows, expected):
    gen_calls = [c for r in rows for c in r['calls'] if c['role'] == 'generation']
    judge_calls = [c for r in rows for c in r['calls'] if c['role'] == 'judge']
    valid_usage = bool(gen_calls) and all(c['usage_verified'] for c in gen_calls)
    n = len(rows)
    return {
        'questions_attempted': n, 'questions_expected': expected, 'full_evaluation': n == expected,
        'generation_success_rate': sum(r['generation_error'] is None for r in rows) / max(n, 1),
        'judge_success_rate': sum(r['judge_error'] is None and r['score'] is not None for r in rows) / max(n, 1),
        'accuracy': sum(bool(r['passing']) for r in rows) / max(n, 1),
        'f1': sum(r['f1'] for r in rows) / max(n, 1),
        'exact_match': sum(r['exact_match'] for r in rows) / max(n, 1),
        'total_tokens_mean': (sum(c['usage'].get('prompt_tokens', 0) + c['usage'].get('completion_tokens', 0)
                                  for c in gen_calls) / n) if valid_usage and n else None,
        'usage_verified': valid_usage,
        'generator_calls': len(gen_calls), 'judge_calls': len(judge_calls),
        'judge_tokens': sum(c['usage'].get('prompt_tokens', 0) + c['usage'].get('completion_tokens', 0)
                            for c in judge_calls),
        'p80_time': percentile([r['generation_seconds'] for r in rows], .8),
        'mean_time': sum(r['generation_seconds'] for r in rows) / max(n, 1),
        'truncated_questions': sum(r['truncated'] for r in rows),
        'generation_errors': sum(r['generation_error'] is not None for r in rows),
        'judge_errors': sum(r['judge_error'] is not None for r in rows),
    }


async def evaluate_candidate(protocol, study, builder, params, pairs, destination, candidate_id):
    from syftr.evaluation.evaluator_factory import CorrectnessEvaluatorFactory
    started = time.monotonic()
    with CallProxy(protocol['server_url'], destination / 'calls.jsonl') as proxy:
        proxy.phase(candidate_id=candidate_id, qa_id=None, role='construction')
        flow = builder.build_flow({**params, 'enforce_full_evaluation': True}, study)
        construction_seconds = time.monotonic() - started
        evaluator = CorrectnessEvaluatorFactory(study).get_evaluators()[0]
        rows = []
        for pair in pairs:
            row = {'candidate_id': candidate_id, 'qa_id': pair.id, 'question': pair.question,
                   'reference': pair.answer, 'qtype': pair.qtype, 'difficulty': pair.difficulty,
                   'answer': '', 'score': None, 'passing': False, 'generation_error': None,
                   'judge_error': None, 'feedback': None, 'truncated': False}
            offset = proxy.phase(candidate_id=candidate_id, qa_id=pair.id, role='generation')
            qstart = time.monotonic()
            try:
                response, _duration, _instrumentation = await asyncio.wait_for(
                    flow.agenerate(pair.question), timeout=protocol['question_timeout'])
                row['answer'] = response.text
                if not row['answer'].strip():
                    raise ValueError('Empty generated answer')
            except Exception as e:
                row['generation_error'] = f'{type(e).__name__}: {e}'
            row['generation_seconds'] = time.monotonic() - qstart
            row.update(answer_scores(row['answer'], pair.answer))
            if row['generation_error']:
                row.update(f1=0.0, exact_match=0.0)
            generation_calls = proxy.since(offset)
            row['truncated'] = any(choice.get('finish_reason') == 'length'
                                   for c in generation_calls for choice in c['response'].get('choices', []))
            proxy.phase(candidate_id=candidate_id, qa_id=pair.id, role='judge')
            jstart = time.monotonic()
            if not row['generation_error']:
                try:
                    judged = await asyncio.wait_for(
                        evaluator.aevaluate(query=pair.question, response=row['answer'], reference=pair.answer),
                        timeout=protocol['request_timeout'])
                    if judged.score is None or not 1 <= judged.score <= 5:
                        raise ValueError('Judge did not return a score in [1,5]')
                    row.update(score=float(judged.score), passing=bool(judged.passing), feedback=judged.feedback)
                except Exception as e:
                    row['judge_error'] = f'{type(e).__name__}: {e}'
            row['judge_seconds'] = time.monotonic() - jstart
            # Requests/responses live in calls.jsonl; compact per-QA usage is also retained.
            row['calls'] = [{k: c[k] for k in ['role', 'http_status', 'elapsed_seconds', 'usage', 'usage_verified']}
                            for c in proxy.since(offset)]
            append_jsonl(destination / 'questions.jsonl', row)
            rows.append(row)
            atomic_json(destination / 'progress.json', summarize(rows, len(pairs)))
            print(f'QUESTION_FINISHED {len(rows)}/{len(pairs)} generation_error={row["generation_error"]} judge_error={row["judge_error"]}', flush=True)
        metrics = summarize(rows, len(pairs))
        metrics.update(construction_seconds=construction_seconds,
                       candidate_seconds=time.monotonic() - started,
                       construction_model_calls=sum(c['role'] == 'construction' for c in proxy.calls))
        # Construction-time LLM calls would require a declared amortization rule.
        if metrics['construction_model_calls']:
            metrics['usage_verified'] = False
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    dest = Path(job['destination'])
    dest.mkdir(parents=True, exist_ok=True)
    random.seed(job['evaluation_seed'])
    try:
        import numpy as np
        np.random.seed(job['evaluation_seed'])
        study, builder, _hf = prepare(job['protocol'], job['workspace'], job['sources'])
        from .device_policy import device_diagnostics
        atomic_json(dest / 'embedding_device.json', device_diagnostics(job['protocol']))
        pairs, manifest = question_manifest(study, job['protocol'])
        if manifest != job['question_manifest']:
            raise ValueError('Worker QA manifest differs from the frozen manifest')
        metrics = asyncio.run(evaluate_candidate(job['protocol'], study, builder, job['params'], pairs,
                                                 dest, job['candidate_id']))
        atomic_json(dest / 'result.json', {'status': 'evaluated', 'metrics': metrics})
    except BaseException as e:
        atomic_json(dest / 'result.json', {'status': 'failed', 'error': f'{type(e).__name__}: {e}',
                                          'traceback': traceback.format_exc()})
        raise
    finally:
        gc.collect()


if __name__ == '__main__':
    main()
