"""Loopback-only OpenAI request journal. Actual endpoint usage, including nested calls.

Streaming is rejected rather than silently losing usage. Evaluation is sequential;
the context is updated only after all requests in the current phase have finished.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import urllib.error
import urllib.request
from uuid import uuid4
from urllib.parse import urlsplit
from .records import append_jsonl


class CallProxy:
    def __init__(self, upstream, path, port=8766):
        parsed = urlsplit(upstream)
        if parsed.hostname not in {'localhost', '127.0.0.1'} or parsed.scheme != 'http':
            raise ValueError('Only local HTTP model endpoints are permitted')
        self.upstream, self.path, self.port = upstream.rstrip('/'), path, port
        self.context, self.calls = {}, []
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                body = json.loads(raw)
                if body.get('stream'):
                    self.send_error(400, 'Streaming disabled for complete usage accounting')
                    return
                if self.path not in {'/v1/chat/completions', '/v1/completions'}:
                    self.send_error(404, 'Unsupported model endpoint')
                    return
                with owner.lock:
                    context = dict(owner.context)
                    call_id = uuid4().hex
                    append_jsonl(owner.path.with_name('requests.jsonl'),
                                 {**context, 'call_id': call_id, 'time': time.time(), 'request': body})
                started = time.monotonic()
                status, error = 200, None
                try:
                    req = urllib.request.Request(owner.upstream + self.path, data=raw,
                                                 headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=600) as response:
                        payload = response.read()
                except urllib.error.HTTPError as e:
                    status, payload, error = e.code, e.read(), str(e)
                except Exception as e:
                    status, error = 502, str(e)
                    payload = json.dumps({'error': {'message': str(e)}}).encode()
                try:
                    decoded = json.loads(payload)
                except json.JSONDecodeError:
                    decoded = {'unparsed_response': payload.decode(errors='replace')}
                usage = decoded.get('usage') or {}
                record = {**context, 'call_id': call_id, 'time': time.time(), 'elapsed_seconds': time.monotonic() - started,
                          'http_status': status, 'error': error, 'request': body,
                          'response': decoded, 'usage': usage,
                          'usage_verified': all(isinstance(usage.get(k), int) and usage[k] >= 0
                                                for k in ['prompt_tokens', 'completion_tokens'])}
                with owner.lock:
                    append_jsonl(owner.path, record)
                    owner.calls.append(record)
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def phase(self, **context):
        with self.lock:
            self.context = context
            return len(self.calls)

    def since(self, offset):
        with self.lock:
            return list(self.calls[offset:])
