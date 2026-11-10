module main

import config
import flag
import os
import api
import infer

// Boot entry: no shell, no REPL — load weights, serve the API.
// Precedence: CLI flag (when != default) > --config file > built-in defaults.
fn main() {
	mut fp := flag.new_flag_parser(os.args)
	cfg_path := fp.string('config', `g`, '', 'path to vllm.json config file (see vllm.example.json)')
	weights := fp.string('model', `m`, 'models/qwen3-1.7b-q4_k_m.gguf', 'path to weights baked into the image')
	port := fp.int('port', `p`, 8000, 'listen port for the OpenAI-compat API')
	ctx_cap := fp.int('ctx', `c`, 4096, 'KV context cap (v1 trims Qwen3 native 32k)')
	fp.finalize() or {
		eprintln(err)
		exit(1)
	}
	mut cc := config.defaults()
	if cfg_path != '' {
		cc = config.load(cfg_path) or {
			eprintln('[vllm-os] bad config: ${err}')
			exit(1)
		}
	}
	// CLI wins when explicitly set (differs from the flag default).
	if weights != 'models/qwen3-1.7b-q4_k_m.gguf' {
		cc.model.weights = weights
	}
	if port != 8000 {
		cc.server.port = port
	}
	if ctx_cap != 4096 {
		cc.model.ctx_cap = ctx_cap
	}
	cc.validate() or {
		eprintln('[vllm-os] bad config: ${err}')
		exit(1)
	}
	cfg := infer.Qwen3Config{
		ctx_cap: cc.model.ctx_cap
	}
	if !os.exists(cc.model.weights) {
		eprintln('[vllm-os] weights not found: ${cc.model.weights}')
		eprintln('[vllm-os] run scripts/fetch_model.sh first, pass --model <file>, or set model.weights in config')
		eprintln('[vllm-os] starting in API-only stub mode (no weights).')
		stub := infer.Model{
			cfg:  cfg
			gguf: infer.GgufModel{
				path: cc.model.weights
			}
		}
		api.serve(cc.server.port, &stub) or {
			eprintln('[vllm-os] serve failed: ${err}')
			exit(1)
		}
		return
	}
	model := infer.load_model(cc.model.weights, cfg) or {
		eprintln('[vllm-os] bad weights: ${err}')
		exit(1)
	}
	eprintln('[vllm-os] boot ok: ${model.cfg.model_id} format=${model.format} tensors=${model.n_tensors()}')
	api.serve(cc.server.port, &model) or {
		eprintln('[vllm-os] serve failed: ${err}')
		exit(1)
	}
}
