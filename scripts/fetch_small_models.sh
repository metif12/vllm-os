#!/usr/bin/env sh
# Host-only tooling. NOT shipped in the OS image.
# Downloads the small GGUF fixtures used by src/infer/gguf_golden_test.v
# and scripts/bench.py. Direct file downloads, no conversion, no login
# (both repos are ungated Apache-2.0; the old Llama fixture needed an HF
# license accept and is gone). The 0.6B tokenizer.json covers the 1.7B
# file too (same 151936 Qwen3 vocab).
# Sizes are asserted after download (corrupt/truncated files fail fast).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
fetch() {
  url="$1"
  out="$2"
  want="$3"
  if [ -f "$out" ] && [ "$(wc -c < "$out")" = "$want" ]; then
    echo "ok: $out"
    return 0
  fi
  echo "fetching $out ..."
  curl -L --fail -o "$out" "$url"
  got="$(wc -c < "$out")"
  if [ "$got" != "$want" ]; then
    echo "size mismatch for $out: want $want, got $got" >&2
    exit 1
  fi
  echo "ok: $out ($got bytes)"
}
fetch "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf" \
  "$ROOT/models/Qwen3-0.6B-Q4_0.gguf" 382156480
fetch "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_0.gguf" \
  "$ROOT/models/Qwen3-1.7B-Q4_0.gguf" 1056782912
