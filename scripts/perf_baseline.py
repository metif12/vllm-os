#!/usr/bin/env python3
"""Python baseline for vllm-os perf comparison. Host-only tooling.
Same shapes, equivalent work as src/infer kernels + the /v1/chat/completions
round trip. Run: python scripts/perf_baseline.py [--kernels] [--api ...].

Scope honesty: this compares (1) kernel throughput and (2) HTTP serving
overhead on identical hardware. It is NOT end-to-end model inference:
the V forward pass (phase 1b) is still pseudo-logits, so a same-model
tokens/sec shootout would be meaningless until both sides run Qwen3-1.7B.
"""
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np


def bench_matvec():
    rng = np.random.default_rng(0)
    w = (rng.integers(0, 13, size=(256 * 512))).astype(np.float32) * 0.125
    x = (rng.integers(0, 7, size=512)).astype(np.float32) * 0.25
    w = w.reshape(256, 512)
    w @ x  # warmup
    t = time.perf_counter()
    for _ in range(20):
        w @ x
    print(f"py matvec 256x512 x20: {(time.perf_counter() - t) * 1e6:.0f}us")

    w2 = (rng.integers(0, 13, size=2048 * 6144)).astype(np.float32).reshape(2048, 6144) * 0.125
    x2 = (rng.integers(0, 7, size=6144)).astype(np.float32) * 0.25
    w2 @ x2  # warmup
    t = time.perf_counter()
    for _ in range(3):
        w2 @ x2
    print(f"py matvec 2048x6144 x3: {(time.perf_counter() - t) * 1e3:.0f}ms")


def numpy_sample(logits, temp=0.8, top_k=40, top_p=0.9, seed=0):
    """Equivalent-work sampler: temp softmax, top-k, top-p truncate, multinomial."""
    rng = np.random.default_rng(seed)
    x = logits / temp
    x = x - x.max()
    e = np.exp(x)
    p = e / e.sum()
    if top_k < p.size:
        part = np.argpartition(p, -top_k)[-top_k:]
        order = part[np.argsort(p[part])][::-1]
    else:
        order = np.argsort(p)[::-1]
    if top_p < 1.0:
        cum = np.cumsum(p[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        order = order[:cut]
    mass = p[order].sum()
    roll = rng.random() * mass
    c = 0.0
    for i in order:
        c += p[i]
        if c >= roll:
            return int(i)
    return int(order[-1])


def bench_sample():
    rng = np.random.default_rng(1)
    logits = -np.arange(64, dtype=np.float32)
    numpy_sample(logits, seed=1)  # warmup
    t = time.perf_counter()
    for i in range(20000):
        logits[i % 64] = -float((i + 7) % 64)
        numpy_sample(logits, seed=1234 + i)
    print(f"py sample 64-logit x20k: {(time.perf_counter() - t) * 1e3:.0f}ms")

    big = -(np.arange(151936, dtype=np.float32) % 1000) * 0.01
    numpy_sample(big, seed=7)  # warmup
    t = time.perf_counter()
    for i in range(20):
        big[(i * 7919) % 151936] = -float(i % 50) * 0.01
        numpy_sample(big, seed=99 + i)
    print(f"py sample 151936-logit x20: {(time.perf_counter() - t) * 1e3:.0f}ms")


_STUB_TEXT = ('qwen3-1.7b tok_in=2 tok_out=64 ids=[35, 34] :: '
               '<|im_start|>user\nopt check<|im_end|>\n<|im_start|>assistant\n')

PY_STUB = r'''
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
BODY = {"id":"cmpl-py","object":"chat.completion","created":1700000000,"model":"qwen3-1.7b",
 "choices":[{"index":0,"message":{"role":"assistant","content":__STUB_JSON__},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":0,"completion_tokens":0}}
PAYLOAD = json.dumps(BODY).encode()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, payload):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        self._send(b'{"object":"list","data":[{"id":"qwen3-1.7b","object":"model","owned_by":"py-stub"}]}')
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(n) if n else b'{}'
        try: req = json.loads(raw)
        except Exception: req = {}
        if self.path == '/v1/chat/completions' and req.get('stream'):
            p = b'data: {"object":"chat.completion.chunk"}\n\ndata: [DONE]\n\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(p)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(p)
        else:
            self._send(PAYLOAD)
HTTPServer(('127.0.0.1', PORT), H).serve_forever()
'''


def post_chat(port, stream=False):
    body = json.dumps({"model": "qwen3-1.7b",
                       "messages": [{"role": "user", "content": "opt check"}],
                       "stream": stream}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
    return (time.perf_counter() - t) * 1e3, len(data)


def bench_api_v(v_server_exe, port=18010, n=200):
    proc = subprocess.Popen([v_server_exe, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                for _ in range(20):
                    post_chat(port)
                break
            except Exception:
                time.sleep(0.1)
        lats = []
        for _ in range(n):
            ms, _ = post_chat(port)
            lats.append(ms)
        return sorted(lats)
    finally:
        proc.terminate()


def bench_api_py(port=18011, n=200):
    code = PY_STUB.replace('__STUB_JSON__', json.dumps(_STUB_TEXT)).replace('PORT', str(port))
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                for _ in range(20):
                    post_chat(port)
                break
            except Exception:
                time.sleep(0.1)
        lats = []
        for _ in range(n):
            ms, _ = post_chat(port)
            lats.append(ms)
        return sorted(lats)
    finally:
        proc.terminate()


def pct(lats, q):
    return lats[min(len(lats) - 1, int(q * len(lats)))]


def main():
    args = set(sys.argv[1:])
    if not args or "--kernels" in args:
        bench_matvec()
        bench_sample()
    if "--api-v" in args or "--api" in args:
        exe = sys.argv[sys.argv.index("--v-exe") + 1] if "--v-exe" in sys.argv else ".\\server.exe"
        lats = bench_api_v(exe)
        print(f"v api chat n={len(lats)} mean={sum(lats)/len(lats):.2f}ms "
              f"min={lats[0]:.2f}ms p50={pct(lats, .5):.2f}ms p99={pct(lats, .99):.2f}ms")
    if "--api-py" in args or "--api" in args:
        lats = bench_api_py()
        print(f"py api chat n={len(lats)} mean={sum(lats)/len(lats):.2f}ms "
              f"min={lats[0]:.2f}ms p50={pct(lats, .5):.2f}ms p99={pct(lats, .99):.2f}ms")


if __name__ == "__main__":
    main()
