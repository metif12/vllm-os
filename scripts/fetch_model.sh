#!/usr/bin/env sh
# Host-only tooling. NOT shipped in the OS image.
# HF safetensors -> GGUF -> Q4_K_M via llama.cpp. Needs python3 + HF login for gated repos.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
OUT="${OUT:-$ROOT/models/qwen3-1.7b-q4_k_m.gguf}"
LLAMA_CPP="${LLAMA_CPP:-$ROOT/third_party/llama.cpp}"
mkdir -p "$ROOT/models" "$ROOT/third_party"
if [ ! -d "$LLAMA_CPP/.git" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP"
fi
if [ ! -f "$ROOT/models/hf/config.json" ]; then
  echo "Downloading $MODEL_ID ..."
  huggingface-cli download "$MODEL_ID" --local-dir "$ROOT/models/hf" --local-dir-use-symlinks False
fi
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$ROOT/models/hf" --outfile "$ROOT/models/qwen3-1.7b-f16.gguf"
cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP/build" -j --target llama-quantize
"$LLAMA_CPP/build/bin/llama-quantize" "$ROOT/models/qwen3-1.7b-f16.gguf" "$OUT" Q4_K_M
ls -lh "$OUT"
