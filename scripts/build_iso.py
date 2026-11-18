#!/usr/bin/env python3
"""Phase 6: guided ISO builder for custom models.

Picks target arch, model weights, and serving config, validates the
combination, then drives the canonical `make image` flow. Run from WSL:

  python3 scripts/build_iso.py                  # interactive
  python3 scripts/build_iso.py --model path/to/foo.gguf --port 8000
  python3 scripts/build_iso.py --dry-run        # print, don't build

What it checks before spending minutes on a 400 MB image:
  * arch: only x86_64 boots today (arm64 is refused loudly, not silently
    misbuilt — the fork's virtio/e1000 drivers are amd64-only);
  * model file exists; weights + ~1 GiB runtime headroom must fit the
    guest RAM budget (--mem, default 4 GiB);
  * tokenizer.json sits next to the weights (real BPE; --allow-stub to
    boot without one);
  * port/ctx/backend/preset/temp ranges mirror cfg.validate, so typos
    fail here instead of at guest boot.

Config: --config FILE bakes that file as-is; otherwise a config.json is
generated from the knobs (baked weights path is always
/models/model.gguf — the fixed image layout).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHES = ("x86_64",)
BACKENDS = ("cpu", "cpu-blocked", "cpu-simd", "cpu-parallel", "auto")
PRESETS = ("deterministic", "precise", "balanced", "creative")
OVERHEAD_BYTES = 1 << 30


def fail(msg):
    print(f"build_iso: error: {msg}", file=sys.stderr)
    return 2


def prompt(text, default, non_interactive):
    if non_interactive or not sys.stdin.isatty():
        if default is None:
            return None
        print(f"{text} [{default}]: {default} (non-interactive)")
        return default
    try:
        v = input(f"{text} [{default}]: ").strip()
    except EOFError:
        v = ""
    return v if v else default


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="x86_64",
                    help="target arch (only x86_64 boots today)")
    ap.add_argument("--model", default=None, help="GGUF weights file")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json (default: next to --model)")
    ap.add_argument("--allow-stub-tokenizer", action="store_true",
                    help="boot without a tokenizer (stub <tok_N> ids)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--config", default=None,
                    help="bake this config file as-is (skips generation)")
    ap.add_argument("--mem", type=int, default=4,
                    help="guest RAM budget in GiB")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--v", default=None)
    ap.add_argument("--non-interactive", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the make invocation, don't build")
    return ap.parse_args(argv)


def resolve(args):
    if args.arch not in ARCHES:
        return fail(f"arch {args.arch!r} cannot boot yet "
                    f"(supported: {', '.join(ARCHES)}); refusing")
    model = args.model
    if not model:
        model = prompt("GGUF weights file", None, args.non_interactive)
        if not model:
            return fail("no --model given (and no tty to ask)")
    if not os.path.isfile(model):
        return fail(f"model not found: {model}")
    model = os.path.abspath(model)
    tok = args.tokenizer
    if not tok:
        tok = os.path.join(os.path.dirname(model), "tokenizer.json")
    if not os.path.isfile(tok):
        if args.allow_stub_tokenizer:
            print("warning: no tokenizer.json, guest boots with stub ids")
            tok = None
        else:
            return fail(f"tokenizer.json not found: {tok} "
                        "(--allow-stub-tokenizer to boot without one)")
    else:
        tok = os.path.abspath(tok)
    if not 1 <= args.port <= 65535:
        return fail(f"port out of range: {args.port}")
    if not 128 <= args.ctx <= 131072:
        return fail(f"ctx out of range: {args.ctx}")
    if args.backend not in BACKENDS:
        return fail(f"unknown backend: {args.backend} "
                    f"(one of {', '.join(BACKENDS)})")
    if args.preset not in PRESETS:
        return fail(f"unknown sampler preset: {args.preset} "
                    f"(one of {', '.join(PRESETS)})")
    if args.temp is not None and not 0.0 <= args.temp <= 2.0:
        return fail(f"temp out of range: {args.temp}")
    if args.max_tokens is not None and args.max_tokens < 1:
        return fail(f"max_tokens must be >= 1: {args.max_tokens}")
    size = os.path.getsize(model)
    budget = args.mem * (1 << 30)
    if size + OVERHEAD_BYTES > budget:
        return fail(f"weights ({size // (1 << 20)} MiB + 1 GiB runtime) "
                    f"exceed the {args.mem} GiB guest budget (--mem)")
    print(f"model: {model} ({size // (1 << 20)} MiB)")
    print(f"tokenizer: {tok if tok else '(stub)'}")
    return args, model, tok


def gen_config(args):
    sampler = {"preset": args.preset}
    if args.temp is not None:
        sampler["temp"] = args.temp
    if args.top_k is not None:
        sampler["top_k"] = args.top_k
    if args.top_p is not None:
        sampler["top_p"] = args.top_p
    api = {}
    if args.max_tokens is not None:
        api["max_tokens"] = args.max_tokens
    return {
        "server": {"host": "0.0.0.0", "port": args.port},
        "model": {"weights": "/models/model.gguf", "format": "auto",
                  "ctx_cap": args.ctx, "backend": args.backend},
        "sampler": sampler,
        "api": api,
    }


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    r = resolve(args)
    if not isinstance(r, tuple):
        return r
    args, model, tok = r
    cfg_path = None
    if args.config:
        if not os.path.isfile(args.config):
            return fail(f"config not found: {args.config}")
        cfg_path = os.path.abspath(args.config)
        print(f"config: {cfg_path} (as-is)")
    else:
        work = (args.work if args.work
                else os.path.join(ROOT, "build", "image-work"))
        os.makedirs(work, exist_ok=True)
        cfg_path = os.path.join(os.path.abspath(work), "iso-config.json")
        with open(cfg_path, "w") as f:
            json.dump(gen_config(args), f, indent=2)
            f.write("\n")
        print(f"config: {cfg_path} (generated)")

    cmd = ["make", "-C", ROOT,
           "-f", "GNUmakefile", "image", f"MODEL={model}"]
    if tok:
        cmd.append(f"TOKENIZER={tok}")
    cmd.append(f"CONFIG={cfg_path}")
    for name, val in (("KERNEL", args.kernel), ("OUT", args.out),
                      ("WORK", args.work), ("EXTRA", args.extra),
                      ("V", args.v)):
        if val:
            cmd.append(f"{name}={val}")
    print("+ " + " ".join(cmd))
    if args.dry_run:
        return 0
    try:
        r = subprocess.run(cmd)
    except OSError as e:
        print(f"build_iso: error: cannot run make ({e}); "
              "run this script from WSL", file=sys.stderr)
        return 1
    if r.returncode != 0:
        print("build_iso: make image failed", file=sys.stderr)
        return 1
    out = args.out if args.out else os.path.join(ROOT, "vllm-os.iso")
    print(f"image ready: {out}")
    print("boot it: qemu-system-x86_64 -cdrom "
          f"{out} -m {args.mem}G -display none -serial stdio "
          f"-netdev user,id=n0,hostfwd=tcp::18000-:{args.port} "
          "-device e1000,netdev=n0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
