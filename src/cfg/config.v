module cfg

import infer
import json2
import os

// File-based configuration: `vllm.json` (or --config path) carries every
// knob; environment and CLI flags override individual fields. Precedence:
// CLI flag (when != default) > env `VLLM_*` > config file > built-in
// defaults. Partial files are valid — missing fields normalize to
// defaults, then the whole config validates. A UTF-8 BOM (Notepad)
// is stripped first.

pub struct ServerCfg {
pub mut:
	host      string = '0.0.0.0'
	port      int    = 8000
	max_conns int    = 64
	max_body  int    = 1 << 20
}

pub struct ModelCfg {
pub mut:
	weights string = 'models/qwen3-1.7b-q4_k_m.gguf'
	format  string = 'auto' // auto | gguf | safetensors
	ctx_cap int    = 4096
	backend string = 'cpu' // cpu | cpu-blocked | cpu-simd | auto
}

pub struct SamplerCfg {
pub mut:
	preset string = 'balanced' // deterministic | precise | balanced | creative
	temp   f32    = 0.7
	top_k  int    = 40
	top_p  f32    = 0.9
	min_p  f32    = 0.0 // 0 = disabled; else drop tokens below min_p * top-1 prob
	seed   u64
	fast   bool // approx-softmax fast path (sampling only)
}

pub struct ApiCfg {
pub mut:
	max_tokens int = 256
}

pub struct Config {
pub mut:
	server  ServerCfg
	model   ModelCfg
	sampler SamplerCfg
	api     ApiCfg
}

// defaults returns the built-in configuration (used when no --config file).
pub fn defaults() Config {
	return Config{}
}

// load reads JSON config and normalizes zeros/empties back to defaults,
// so partial files ("just set the port") work as expected.
// A UTF-8 BOM (common from Windows editors) is stripped first.
pub fn load(path string) !Config {
	mut raw := os.read_file(path)!
	// Strip UTF-8 BOM by bytes: V string escapes don't cover \ufeff portably.
	if raw.len >= 3 && raw[0] == 0xEF && raw[1] == 0xBB && raw[2] == 0xBF {
		raw = raw[3..]
	}
	mut c := json2.decode[Config](raw)!
	c.normalize()
	c.validate()!
	return c
}

// normalize fills zero/empty fields back to defaults so partial files work.
pub fn (mut c Config) normalize() {
	if c.server.port == 0 {
		c.server.port = 8000
	}
	if c.server.host == '' {
		c.server.host = '0.0.0.0'
	}
	if c.server.max_conns == 0 {
		c.server.max_conns = 64
	}
	if c.server.max_body == 0 {
		c.server.max_body = 1 << 20
	}
	if c.model.weights == '' {
		c.model.weights = 'models/qwen3-1.7b-q4_k_m.gguf'
	}
	if c.model.format == '' {
		c.model.format = 'auto'
	}
	if c.model.ctx_cap == 0 {
		c.model.ctx_cap = 4096
	}
	if c.model.backend == '' {
		c.model.backend = 'cpu'
	}
	if c.sampler.preset == '' {
		c.sampler.preset = 'balanced'
	}
	if c.api.max_tokens == 0 {
		c.api.max_tokens = 256
	}
}

// validate rejects out-of-range ports, ctx caps, and unknown backend/format names.
pub fn (c Config) validate() ! {
	if c.server.port < 1 || c.server.port > 65535 {
		return error('config: port out of range: ${c.server.port}')
	}
	if c.server.max_conns < 1 || c.server.max_conns > 4096 {
		return error('config: max_conns out of range: ${c.server.max_conns}')
	}
	if c.server.max_body < 4096 || c.server.max_body > 64 << 20 {
		return error('config: max_body out of range: ${c.server.max_body}')
	}
	if c.model.ctx_cap < 128 || c.model.ctx_cap > 131072 {
		return error('config: ctx_cap out of range: ${c.model.ctx_cap}')
	}
	if c.model.backend !in ['cpu', 'cpu-blocked', 'cpu-simd', 'cpu-parallel', 'auto'] {
		return error('config: unknown backend: ${c.model.backend}')
	}
	if c.model.format !in ['auto', 'gguf', 'safetensors'] {
		return error('config: unknown format: ${c.model.format}')
	}
	if c.sampler.preset !in infer.sampler_presets() {
		return error('config: unknown sampler preset: ${c.sampler.preset}')
	}
	if c.sampler.temp < 0.0 || c.sampler.temp > 2.0 {
		return error('config: temp out of range: ${c.sampler.temp}')
	}
	if c.sampler.top_k < 0 || c.sampler.top_k > 200000 {
		return error('config: top_k out of range: ${c.sampler.top_k}')
	}
	if c.sampler.top_p <= 0.0 || c.sampler.top_p > 1.0 {
		return error('config: top_p out of range: ${c.sampler.top_p}')
	}
	if c.sampler.min_p < 0.0 || c.sampler.min_p >= 1.0 {
		return error('config: min_p out of range: ${c.sampler.min_p}')
	}
	if c.api.max_tokens < 1 || c.api.max_tokens > 32768 {
		return error('config: max_tokens out of range: ${c.api.max_tokens}')
	}
}

// sampler_params resolves the effective sampler: preset base with explicit
// non-default fields layered on top. SamplerCfg{} defaults equal the
// balanced preset, so untouched fields never override the preset.
pub fn (c Config) sampler_params() infer.SampleParams {
	mut p := infer.preset_params(c.sampler.preset) or { infer.SampleParams{} }
	d := SamplerCfg{}
	if c.sampler.temp != d.temp {
		p.temp = c.sampler.temp
	}
	if c.sampler.top_k != d.top_k {
		p.top_k = c.sampler.top_k
	}
	if c.sampler.top_p != d.top_p {
		p.top_p = c.sampler.top_p
	}
	if c.sampler.min_p != d.min_p {
		p.min_p = c.sampler.min_p
	}
	if c.sampler.seed != d.seed {
		p.seed = c.sampler.seed
	}
	if c.sampler.fast != d.fast {
		p.fast = c.sampler.fast
	}
	return p
}

// apply_env overrides fields from VLLM_* environment variables. Called
// after load() (file) and before CLI flags, establishing the documented
// CLI > env > file > defaults precedence. Unset vars are skipped;
// malformed values are errors, not silent skips.
pub fn (mut c Config) apply_env() ! {
	if v := os.getenv_opt('VLLM_HOST') {
		c.server.host = v
	}
	if v := os.getenv_opt('VLLM_PORT') {
		c.server.port = v.int()
	}
	if v := os.getenv_opt('VLLM_MAX_CONNS') {
		c.server.max_conns = v.int()
	}
	if v := os.getenv_opt('VLLM_MAX_BODY') {
		c.server.max_body = v.int()
	}
	if v := os.getenv_opt('VLLM_MODEL') {
		c.model.weights = v
	}
	if v := os.getenv_opt('VLLM_FORMAT') {
		c.model.format = v
	}
	if v := os.getenv_opt('VLLM_CTX') {
		c.model.ctx_cap = v.int()
	}
	if v := os.getenv_opt('VLLM_BACKEND') {
		c.model.backend = v
	}
	if v := os.getenv_opt('VLLM_PRESET') {
		c.sampler.preset = v
	}
	if v := os.getenv_opt('VLLM_TEMP') {
		c.sampler.temp = v.f32()
	}
	if v := os.getenv_opt('VLLM_TOP_K') {
		c.sampler.top_k = v.int()
	}
	if v := os.getenv_opt('VLLM_TOP_P') {
		c.sampler.top_p = v.f32()
	}
	if v := os.getenv_opt('VLLM_MIN_P') {
		c.sampler.min_p = v.f32()
	}
	if v := os.getenv_opt('VLLM_SEED') {
		c.sampler.seed = v.u64()
	}
	if v := os.getenv_opt('VLLM_FAST') {
		c.sampler.fast = v == '1' || v.to_lower() == 'true'
	}
	if v := os.getenv_opt('VLLM_MAX_TOKENS') {
		c.api.max_tokens = v.int()
	}
	c.normalize()
	c.validate()!
}
