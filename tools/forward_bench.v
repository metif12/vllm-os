module main

// Forward throughput probe (host-only): prefill + N timed decode steps on
// a real GGUF file. Not gated in CI (needs weights on disk).
// Usage (from repo root):
//   v -prod -cc gcc -path "src|@vlib|@vmodules" -o /tmp/fwdbench tools/forward_bench.v
//   /tmp/fwdbench /path/to/model.gguf [n]
// Prints prefill ms + per-decode ms + best. Threading follows
// VLLM_THREADS (default 4); VLLM_NO_SIMD=1 forces the pure-V path.
// NOTE: print/println only, no interpolation — the release-V -prod
// cgen quirk drops the str_intp runtime for interpolation in tiny
// module-main binaries (see forward parity notes).
import infer
import os
import time

fn main() {
	weights := os.args[1]
	n := if os.args.len > 2 { os.args[2].int() } else { 5 }
	m := infer.load_model(weights, infer.Qwen3Config{}) or { panic(err) }
	// Reuse the boot-cached blob when present; fall back to a direct
	// load for Models built without the cache.
	blob := if m.blob.len > 0 { m.blob } else { infer.load_model_blob(weights) or { panic(err) } }
	mut f := infer.new_forward_ctx(m.cfg, m.gguf, blob) or { panic(err) }
	print('model_hidden=')
	println(m.cfg.hidden)
	print('model_layers=')
	println(m.cfg.layers)
	ids := m.tok.encode('Hello world')
	mut sw := time.new_stopwatch()
	for id in ids {
		infer.embed_gather(blob, m.gguf, m.cfg.hidden, id, mut f.hidden) or { panic(err) }
		f.decode_step() or { panic(err) }
	}
	print('prefill_2tok_ms=')
	println(sw.elapsed().milliseconds())
	mut best := i64(9223372036854775807)
	for _ in 0 .. n {
		sw = time.new_stopwatch()
		f.decode_step() or { panic(err) }
		ms := sw.elapsed().milliseconds()
		print('decode_ms=')
		println(ms)
		if ms < best {
			best = ms
		}
	}
	print('decode_best_ms=')
	println(best)
	println('FORWARD-BENCH-OK')
}
