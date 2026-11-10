module main

import flag
import os
import api
import infer

// Boot entry: no shell, no REPL — load weights, serve the API.
fn main() {
	mut fp := flag.new_flag_parser(os.args)
	weights := fp.string('model', `m`, 'models/qwen3-1.7b-q4_k_m.gguf', 'path to GGUF weights baked into the image')
	port := fp.int('port', `p`, 8000, 'listen port for the OpenAI-compat API')
	ctx_cap := fp.int('ctx', `c`, 4096, 'KV context cap (v1 trims Qwen3 native 32k)')
	fp.finalize() or {
		eprintln(err)
		exit(1)
	}
	cfg := infer.Qwen3Config{
		ctx_cap: ctx_cap
	}
	if !os.exists(weights) {
		eprintln('[vllm-os] weights not found: ${weights}')
		eprintln('[vllm-os] run scripts/fetch_model.sh first, or pass --model <file>.gguf')
		eprintln('[vllm-os] starting in API-only stub mode (no weights).')
		stub := infer.Model{
			cfg:  cfg
			gguf: infer.GgufModel{
				path: weights
			}
		}
		api.serve(port, &stub) or {
			eprintln('[vllm-os] serve failed: ${err}')
			exit(1)
		}
		return
	}
	model := infer.load_model(weights, cfg) or {
		eprintln('[vllm-os] bad weights: ${err}')
		exit(1)
	}
	eprintln('[vllm-os] boot ok: ${model.cfg.model_id} tensors=${model.gguf.meta.n_tensors} kv=${model.gguf.meta.n_kv}')
	api.serve(port, &model) or {
		eprintln('[vllm-os] serve failed: ${err}')
		exit(1)
	}
}
