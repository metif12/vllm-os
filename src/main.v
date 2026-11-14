module main

import cfg
import flag
import formats
import infer
import json
import os
import api

// Boot entry: no shell, no REPL — load weights, serve the API.
// Precedence: CLI flag (when != default) > VLLM_* env > --config file >
// built-in defaults. Introspection flags (--list-backends, --list-formats,
// --print-config) print and exit before any weights are touched.
fn main() {
	mut fp := flag.new_flag_parser(os.args)
	cfg_path := fp.string('config', `g`, '',
		'path to vllm.json config file (see vllm.example.json)')
	weights := fp.string('model', `m`, 'models/qwen3-1.7b-q4_k_m.gguf',
		'path to weights baked into the image')
	port := fp.int('port', `p`, 8000, 'listen port for the OpenAI-compat API')
	ctx_cap := fp.int('ctx', `c`, 4096, 'KV context cap (v1 trims Qwen3 native 32k)')
	backend := fp.string('backend', `b`, 'cpu',
		'compute backend: cpu | cpu-blocked | cpu-simd | auto')
	list_backends := fp.bool('list-backends', 0, false, 'print backend registry and exit')
	list_formats := fp.bool('list-formats', 0, false, 'print weight-format handlers and exit')
	print_config := fp.bool('print-config', 0, false, 'print effective config as JSON and exit')
	fp.finalize() or {
		eprintln(err)
		exit(1)
	}
	if list_backends {
		for name in infer.list_backends() {
			be := infer.select_backend(name, 2048, 6144)
			println('${name}\tresolves=${be.name()}')
		}
		return
	}
	if list_formats {
		println('gguf\tGGUF container, native rich index (infer)')
		for h in formats.default_handlers() {
			println('${h.format()}\tregistry handler')
		}
		return
	}
	mut cc := cfg.defaults()
	if cfg_path != '' {
		cc = cfg.load(cfg_path) or {
			eprintln('[vllm-os] bad config: ${err}')
			exit(1)
		}
	}
	// Env layer: VLLM_* overrides file, CLI overrides env.
	cc.apply_env() or {
		eprintln('[vllm-os] bad env: ${err}')
		exit(1)
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
	if backend != 'cpu' {
		cc.model.backend = backend
	}
	cc.validate() or {
		eprintln('[vllm-os] bad config: ${err}')
		exit(1)
	}
	if print_config {
		println(json.encode(cc))
		return
	}
	// Resolve once at boot so a bad backend name fails here, not mid-serve.
	// The phase-1b forward calls select_backend per layer shape; today the
	// stub generate path exercises sampler/tokenizer while the value is
	// validated + logged.
	boot_be := infer.select_backend(cc.model.backend, 2048, 6144)
	eprintln('[vllm-os] backend=${cc.model.backend} resolved=${boot_be.name()} simd=${infer.has_avx2()} threads=${infer.parallel_threads()}')
	qcfg := infer.Qwen3Config{
		ctx_cap: cc.model.ctx_cap
	}
	if !os.exists(cc.model.weights) {
		eprintln('[vllm-os] weights not found: ${cc.model.weights}')
		eprintln('[vllm-os] run scripts/fetch_model.sh first, pass --model <file>, or set model.weights in config')
		eprintln('[vllm-os] starting in API-only stub mode (no weights).')
		stub := infer.Model{
			cfg:  qcfg
			gguf: infer.GgufModel{
				path: cc.model.weights
			}
			tok:  infer.Tokenizer(infer.StubTokenizer{})
		}
		api.serve(cc, &stub) or {
			eprintln('[vllm-os] serve failed: ${err}')
			exit(1)
		}
		return
	}
	model := infer.load_model(cc.model.weights, qcfg) or {
		eprintln('[vllm-os] bad weights: ${err}')
		exit(1)
	}
	eprintln('[vllm-os] boot ok: ${model.cfg.model_id} format=${model.format} tensors=${model.n_tensors()}')
	api.serve(cc, &model) or {
		eprintln('[vllm-os] serve failed: ${err}')
		exit(1)
	}
}
