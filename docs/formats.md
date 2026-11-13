# Weight formats

`src/formats` owns containers behind the `ContainerHandler` registry
(`registry.v`); `infer.load_model` keeps its native GGUF path and
dispatches everything else through `handler_for`, so inference code
never sniffs bytes and format #4 touches no existing dispatch.

## Supported

| container | detection | loader | notes |
|---|---|---|---|
| GGUF `.gguf` | `GGUF` magic | `src/infer/gguf.v` (header + KV + tensor table) | full Q4_K_M index; blobs stay on disk / get mmap'd; native path (rich index) |
| SafeTensors `.safetensors` | u64 header len + `{"` | `SafetensorsHandler.load_index` (header-only) | dtype/shape/data_offsets per tensor, `__metadata__` kept |
| PyTorch `.bin`/`.pt`/`.pth` | extension fallback | `PytorchHintHandler` (clear error) | convert first, see below |

`detect()` reads at most 16 bytes. Registry loaders read only the
JSON index (capped at 256 MB); weight blobs are never pulled into RAM by
the loader. The unified view is `ModelFile{tensors: []TensorMeta, meta}`
with `find(name)`; `Model.n_tensors()` works across containers.
`./server --list-formats` prints the registry without serving.

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
2. Implement `ContainerHandler` (`format` + `sniff_prefix` + `load_index`
   returning `TensorMeta` rows) in `src/formats/registry.v`.
3. Append one line to `default_handlers()` — dispatch needs no edits.
4. Add a fixture test in `src/formats/formats_test.v` (craft bytes
   in-test, no downloads — see the GGUF/SafeTensors fixture tests).
5. Document the row in the table above.
