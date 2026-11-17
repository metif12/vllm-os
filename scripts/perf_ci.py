#!/usr/bin/env python3
"""One-command perf track for CI (and humans): fetch weights, run the
benches, rewrite only the marked CI section of docs/perf.md.

  python scripts/perf_ci.py [--out bench/ci] [--reps 2]
      [--skip-forward] [--run-url URL]

What it does, in order (aborts loudly on the first failure, never
records partial numbers):
  1. ensures models/Qwen3-0.6B-Q4_0.gguf exists at the asserted size
     (downloads it when missing) plus its tokenizer.json sidecar;
  2. runs scripts/bench.py --force (shared CI boxes are always "loaded")
     with the kernel bench, NumPy baseline and --forward probe;
  3. runs scripts/perf_update_md.py on the produced results.json.

Writes bench/<out> (gitignored) and rewrites docs/perf.md in place.
The workflow commits docs/perf.md with [skip ci]; raw results.json
goes to the workflow artifact, not the repo.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL = ("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/"
             "Qwen3-0.6B-Q4_0.gguf")
MODEL_SIZE = 382156480
TOKENIZER_URL = ("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/"
                 "tokenizer.json")


def log(msg):
    print(msg, flush=True)


def ensure_file(url, path, want_size=None, must_contain=None):
    if (os.path.exists(path) and (want_size is None
                                  or os.path.getsize(path) == want_size)):
        log(f"ok: {path}")
        return
    log(f"fetching {path} ...")
    urllib.request.urlretrieve(url, path)
    got = os.path.getsize(path)
    if want_size is not None and got != want_size:
        raise SystemExit(f"size mismatch for {path}: want {want_size}, "
                         f"got {got}")
    if must_contain is not None:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(1 << 20)
        if must_contain not in head:
            raise SystemExit(f"{path} does not look like a tokenizer: "
                             f"missing {must_contain!r}")
    log(f"ok: {path} ({got} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join("bench", "ci"),
                    help="results dir (gitignored)")
    ap.add_argument("--reps", type=int, default=2,
                    help="process repeats per timed section")
    ap.add_argument("--model", default="models/Qwen3-0.6B-Q4_0.gguf",
                    help="GGUF file for the forward bench")
    ap.add_argument("--forward-tokens", type=int, default=5,
                    help="timed decode steps per forward mode")
    ap.add_argument("--skip-forward", action="store_true",
                    help="kernels only (no weights download)")
    ap.add_argument("--run-url", default="",
                    help="workflow run URL for perf.md provenance")
    ap.add_argument("--doc", default=os.path.join(ROOT, "docs", "perf.md"))
    args = ap.parse_args()

    model = (args.model if os.path.isabs(args.model)
             else os.path.join(ROOT, args.model))
    if not args.skip_forward:
        ensure_file(MODEL_URL, model, want_size=MODEL_SIZE)
        ensure_file(TOKENIZER_URL,
                    os.path.join(os.path.dirname(model), "tokenizer.json"),
                    must_contain='"vocab"')

    outdir = (args.out if os.path.isabs(args.out)
              else os.path.join(ROOT, args.out))
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "bench.py"),
           "--force", "--reps", str(args.reps), "--out", outdir]
    if not args.skip_forward:
        cmd += ["--forward", "--model", model,
                "--forward-tokens", str(args.forward_tokens)]
    log("+ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"bench.py failed: {r.returncode}")

    results = os.path.join(outdir, "results.json")
    cmd = [sys.executable, os.path.join(ROOT, "scripts",
                                        "perf_update_md.py"),
           results, "--doc", args.doc]
    if args.run_url:
        cmd += ["--run-url", args.run_url]
    log("+ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"perf_update_md.py failed: {r.returncode}")

    with open(results) as f:
        n = len(json.load(f)["records"])
    log(f"tracked {n} raw records; docs/perf.md CI section updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
