# Weight formats

`src/formats` owns containers; `infer.load_model` dispatches on
`formats.detect()` and inference code never sniffs bytes.

## Supported

| container | detection | loader | notes |
|---|---|---|---|
| GGUF `.gguf` | `GGUF` magic | `src/infer/gguf.v` (header + KV + tensor table) | full Q4_K_M index; blobs stay on disk / get mmap'd |
| SafeTensors `.safetensors` | u64 header len + `{"` | `src/formats/formats.v` (header-only) | dtype/shape/data_offsets per tensor, `__metadata__` kept |
| PyTorch `.bin`/`.pt`/`.pth` | extension fallback | — (clear error) | convert first, see below |

`detect()` reads at most 16 bytes. `load_safetensors()` reads only the
JSON index (capped at 256 MB); weight blobs are never pulled into RAM by
the loader. The unified view is `ModelFile{tensors: []TensorMeta, meta}`
with `find(name)`; `Model.n_tensors()` works across containers.

## Getting Qwen3-1.7B weights

```sh
./scripts/fetch_model.sh   # HF safetensors → GGUF → Q4_K_M into models/
```

Needs host Python + `huggingface-cli` + `llama.cpp` checkout — host-only
tooling, never shipped in the image. The ISO build
(`scripts/build_image.sh`) bakes `models/model.gguf` + `tokenizer.json`
into the boot image; PyTorch checkpoints must be converted the same way
first (the loader error says exactly this).

## Adding a new format

1. Add a `WeightFormat` variant in `src/formats/formats.v`.
2. Teach `detect()` its magic (keep it to a prefix sniff).
3. Write `load_<fmt>() -> !ModelFile` returning `TensorMeta` rows.
4. Add one dispatch arm in `infer.load_model` + a fixture test in
   `src/formats/formats_test.v` (craft bytes in-test, no downloads —
   see the GGUF/SafeTensors fixture tests).
5. Document the row in the table above.
