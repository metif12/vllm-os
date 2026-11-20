#!/usr/bin/env sh
# Host-only tooling. NOT shipped in the OS image.
# Downloads the small GGUF fixtures used by src/infer/gguf_golden_test.v
# and scripts/bench.py. Direct file downloads, no conversion, no login
# (both repos are ungated Apache-2.0; the old Llama fixture needed an HF
# license accept and is gone). The 0.6B tokenizer.json covers the 1.7B
# file too (same 151936 Qwen3 vocab).
# Sizes are asserted after download (corrupt/truncated files fail fast).
# Usage: sh scripts/fetch_small_models.sh [default|q4_k_m|all]
#   default (no args) - the two Q4_0 fixtures, as before
#   q4_k_m            - only the Phase 11 target Q4_K_M fixture
#   all               - both sets
# Curl honours http_proxy/https_proxy from the environment as before.
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
fetch_q4_0() {
  fetch "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf" \
    "$ROOT/models/Qwen3-0.6B-Q4_0.gguf" 382156480
  fetch "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_0.gguf" \
    "$ROOT/models/Qwen3-1.7B-Q4_0.gguf" 1056782912
}
fetch_q4_k_m() {
  # Exact size from the HF blob metadata for this repo/file:
  # GET /api/models/unsloth/Qwen3-1.7B-GGUF?blobs=true, LFS size 1107409472,
  # sha256 b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897
  # (checked 2026-11-20). A truncated download therefore fails the same byte
  # assertion as the Q4_0 fixtures; curl -L --fail also rejects HTTP errors.
  fetch "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf" \
    "$ROOT/models/Qwen3-1.7B-Q4_K_M.gguf" 1107409472
}

# Default (no argument) preserves the legacy Q4_0-only invocation.
case "${1:-default}" in
  default)
    fetch_q4_0
    ;;
  q4_k_m)
    fetch_q4_k_m
    ;;
  all)
    fetch_q4_0
    fetch_q4_k_m
    ;;
  *)
    echo "usage: $0 [default|q4_k_m|all]" >&2
    exit 2
    ;;
esac
