"""Visible readiness checks before consuming a study budget."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request
from .records import atomic_json, environment


def request(base, body):
    req = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def check_server(base):
    common = {'model': 'rm-qwen25-7b-q4', 'temperature': 0, 'max_tokens': 256, 'stream': False}
    plain = request(base, {**common, 'messages': [{'role': 'user', 'content': 'Reply with the single word READY.'}]})
    tool = request(base, {**common, 'messages': [{'role': 'user', 'content': 'Call lookup with query Paris.'}],
                          'tools': [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Look up a fact',
                              'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}],
                          'tool_choice': {'type': 'function', 'function': {'name': 'lookup'}}})
    calls = tool.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
    if not calls or calls[0]['function']['name'] != 'lookup':
        raise ValueError('Tool calling did not return the expected lookup function')
    json.loads(calls[0]['function']['arguments'])
    for result in [plain, tool]:
        usage = result.get('usage', {})
        if not all(isinstance(usage.get(k), int) for k in ['prompt_tokens', 'completion_tokens']):
            raise ValueError('Server does not return usable token counts')
    # Mechanical calibration, not a substitute for human judge validation.
    judgments = []
    from .adapter import JUDGE_SYSTEM
    for answer, expected in [('Paris', True), ('London', False)]:
        response = request(base, {**common, 'messages': [
            {'role': 'system', 'content': JUDGE_SYSTEM},
            {'role': 'user', 'content': f'Question: What is the capital of France? Reference: Paris. Answer: {answer}'}]})
        text = response['choices'][0]['message']['content']
        parsed = json.loads(text)
        score = parsed.get('score')
        observed = (score >= 4) if isinstance(score, (int, float)) and 1 <= score <= 5 else None
        judgments.append({'answer': answer, 'expected': expected, 'observed': observed, 'response': response})
        if observed is not expected:
            raise ValueError('Basic local judge calibration failed')
    return {'status': 'passed', 'completion': plain, 'tool_call': tool, 'mechanical_judge_check': judgments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--server-url', default='http://127.0.0.1:8765')
    args = parser.parse_args()
    record = {'environment': environment(), 'status': 'running', 'time': time.time()}
    try:
        if not (3, 11, 1) <= sys.version_info[:3] < (3, 13):
            raise RuntimeError('Syftr requires Python >=3.11.1,<3.13')
        import optuna
        if optuna.__version__ != '4.2.1':
            raise RuntimeError('Restore pinned Optuna 4.2.1')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU unavailable')
        record['server'] = check_server(args.server_url)
        record['status'] = 'passed'
    except Exception as e:
        record.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        atomic_json(args.output, record)


if __name__ == '__main__':
    main()
