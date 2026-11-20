module main

import cfg
import flag
import formats
import infer
import json
import os
import api
import runtime

// print_tune is a host-only introspection flag: detect CPU/RAM and print a
// concrete backend/threads/quant recommendation. It never starts a server.
fn print_tune() {
	cpus := runtime.nr_cpus()
	total_mb := runtime.total_memory() or { 0 } / (1024 * 1024)
	free_mb := runtime.free_memory() or { 0 } / (1024 * 1024)
	avx2 := infer.has_avx2()
	// The measured default (docs/perf.md) saturates DRAM at four streaming
	// workers; VLLM_THREADS overrides it per host.
	threads := infer.parallel_threads()
	backend := if avx2 { 'auto' } else { 'cpu-blocked' }
	// Q4_0 is the quant with fused AVX2 kernels and measured tok/s in-tree.
	quant := 'Q4_0'
	// Keep a 4 GB guest budget: weights + KV + server.
	ctx_cap := if total_mb > 0 && total_mb < 4096 { 2048 } else { 4096 }
	eprintln('[vllm-os] tune cpus=${cpus} avx2=${avx2} total_mb=${total_mb} free_mb=${free_mb}')
	println('backend=${backend}')
	println('threads=${threads}')
	println('quant=${quant}')
	println('ctx_cap=${ctx_cap}')
}

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
		'compute backend: cpu | cpu-blocked | cpu-simd | cpu-parallel | auto')
	auth_token := fp.string('auth-token', 0, '', 'bearer token (empty = open mode)')
	rate_per_sec := fp.int('rate-per-sec', 0, 0, 'per-IP req/s cap (0 = off)')
	rate_burst := fp.int('rate-burst', 0, 0, 'per-IP burst (0 = one second worth)')
	list_backends := fp.bool('list-backends', 0, false, 'print backend registry and exit')
	list_formats := fp.bool('list-formats', 0, false, 'print weight-format handlers and exit')
	print_config := fp.bool('print-config', 0, false, 'print effective config as JSON and exit')
	tune := fp.bool('tune', 0, false, 'print host-tuned backend/threads/quant and exit')
	fp.finalize() or {
		eprintln(err)
		exit(1)
	}
	if tune {
		print_tune()
		return
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
	// Security knobs: CLI wins when explicitly set (non-default), same
	// rule as --port/--model above; env (VLLM_AUTH_TOKEN/VLLM_API_KEY,
	// VLLM_RATE_*) already applied in apply_env.
	if auth_token != '' {
		cc.server.auth_token = auth_token
	}
	if rate_per_sec != 0 {
		cc.server.rate_per_sec = rate_per_sec
	}
	if rate_burst != 0 {
		cc.server.rate_burst = rate_burst
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
	mut model := infer.load_model(cc.model.weights, qcfg) or {
		eprintln('[vllm-os] bad weights: ${err}')
		exit(1)
	}
	// One boot-wide pool for the decode fork-joins; nil when pinned to one
	// thread (VLLM_THREADS=1) or under -freestanding, so the guest stays
	// single-threaded. generate_real_cached copies it into each ForwardCtx.
	model.pool = infer.new_worker_pool(infer.parallel_threads())
	model.pld = cc.pld_config()
	if cc.model.prefix_cache {
		model.prefix = infer.new_prefix_cache()
	}
	eprintln('[vllm-os] boot ok: ${model.cfg.model_id} format=${model.format} tensors=${model.n_tensors()}')
	api.serve(cc, &model) or {
		eprintln('[vllm-os] serve failed: ${err}')
		exit(1)
	}
}
