"""Test-only stand-in for sglang.launch_server / serve_arm_sglang: like sglang, a 'sglang' server warms itself up with an
HTTP request to its own 127.0.0.1 port using the process's proxy environment (urllib honours http_proxy/no_proxy as
requests does), and answers /server_info only once that warmup succeeded (the behaviour observed in jobs 395121/395122,
where /server_info never answered while the warmup sat on the cluster squid proxy)."""
import argparse, json, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ap = argparse.ArgumentParser(); ap.add_argument('--kind', required=True); ap.add_argument('--port', type=int, required=True)
ap.add_argument('--model', required=True); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
ready = threading.Event()
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def reply(self, value, code=200):
        data = json.dumps(value).encode(); self.send_response(code); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == '/get_model_info': return self.reply(dict(model_path=a.model))
        if self.path == '/health': return self.reply(dict(mode='selection', model=a.model, adapter=None))
        if self.path == '/server_info':
            if not ready.wait(60): return self.reply(dict(error='not ready'), 503)
            return self.reply(dict(random_seed=a.seed, sampling_backend='flashinfer', mem_fraction_static=.45, cuda_graph_max_bs=48, context_length=32768))
        self.reply({}, 404)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/generate': return self.reply(dict(text='OK', meta_info=dict(output_token_logprobs=[[0.0, 1, None]], finish_reason={'type': 'stop'}, prompt_tokens=20)))
        if self.path == '/select': return self.reply(dict(raw='{"selection": 1}', input_tokens=10, seconds=.01))
        self.reply({}, 404)
server = ThreadingHTTPServer(('127.0.0.1', a.port), H)
def warmup():
    for _ in range(20):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{a.port}/get_model_info', timeout=5).read(); ready.set(); return
        except Exception:
            time.sleep(.5)
if a.kind == 'sglang': threading.Thread(target=warmup, daemon=True).start()
else: ready.set()
server.serve_forever()
