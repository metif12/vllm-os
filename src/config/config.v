module config

import json2
import os

// File-based configuration: `vllm.json` (or --config path) carries every
// knob; CLI flags override individual fields. Precedence:
// CLI flag (when != default) > config file > built-in defaults.

pub struct ServerCfg {
pub mut:
	host string = '0.0.0.0'
	port int    = 8000
}

pub struct ModelCfg {
pub mut:
	weights string = 'models/qwen3-1.7b-q4_k_m.gguf'
	format  string = 'auto' // auto | gguf | safetensors
	ctx_cap int    = 4096
	backend string = 'cpu' // cpu | cpu-blocked
}

pub struct ApiCfg {
pub mut:
	max_tokens int = 256
}

pub struct Config {
pub mut:
	server ServerCfg
	model  ModelCfg
	api    ApiCfg
}

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

pub fn (mut c Config) normalize() {
	if c.server.port == 0 {
		c.server.port = 8000
	}
	if c.server.host == '' {
		c.server.host = '0.0.0.0'
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
	if c.api.max_tokens == 0 {
		c.api.max_tokens = 256
	}
}

pub fn (c Config) validate() ! {
	if c.server.port < 1 || c.server.port > 65535 {
		return error('config: port out of range: ${c.server.port}')
	}
	if c.model.ctx_cap < 128 || c.model.ctx_cap > 131072 {
		return error('config: ctx_cap out of range: ${c.model.ctx_cap}')
	}
	if c.model.backend !in ['cpu', 'cpu-blocked'] {
		return error('config: unknown backend: ${c.model.backend}')
	}
	if c.model.format !in ['auto', 'gguf', 'safetensors'] {
		return error('config: unknown format: ${c.model.format}')
	}
	if c.api.max_tokens < 1 || c.api.max_tokens > 32768 {
		return error('config: max_tokens out of range: ${c.api.max_tokens}')
	}
}
