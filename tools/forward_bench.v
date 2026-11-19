module main

// Forward throughput probe (host-only): prefill + N timed decode steps on
// a real GGUF file. Not gated in CI (needs weights on disk).
// Usage (from repo root):
//   v -prod -cc gcc -path "src|@vlib|@vmodules" -o /tmp/fwdbench tools/forward_bench.v
//   /tmp/fwdbench /path/to/model.gguf [n]
// Prints prefill ms + per-decode ms + best. Threading follows
// VLLM_THREADS (default 4); VLLM_NO_SIMD=1 forces the pure-V path.
// VLLM_NO_POOL=1 forces the legacy spawn-per-call path (A/B vs the
// boot-wide persistent pool the server uses by default).
// VLLM_PLD=1 reports prompt-lookup draft accept rate (greedy, serial verify;
// correctness gate only — see docs/perf.md for why there is no speed row).
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
	if os.getenv('VLLM_NO_POOL') == '' {
		f.pool = infer.new_worker_pool(infer.parallel_threads())
		print('pool=on')
	} else {
		print('pool=off')
	}
	println('')
	print('model_hidden=')
	println(m.cfg.hidden)
	print('model_layers=')
	println(m.cfg.layers)
	probe := os.getenv('VLLM_PROBE')
	if probe == '' {
		probe = 'Hello world'
	}
	ids := m.tok.encode(probe)
	pld_on := os.getenv('VLLM_PLD') == '1'
	mut ngram := 3
	if v := os.getenv_opt('VLLM_PLD_NGRAM') {
		ngram = v.int()
	}
	mut cont := 10
	if v := os.getenv_opt('VLLM_PLD_CONTINUATION') {
		cont = v.int()
	}
	mut pld := infer.new_pld_controller(infer.PldCfg{
		enabled:      pld_on
		max_ngram:    ngram
		continuation: cont
	}, ids)
	mut emitted := []int{len: cont + 1}
	print('pld=')
	if pld_on {
		println('on')
	} else {
		println('off')
	}
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
		if pld_on {
			got := pld.next_tokens(mut f, blob, m.gguf, mut emitted) or { panic(err) }
			print('pld_tokens=')
			println(got)
		} else {
			f.decode_step() or { panic(err) }
		}
		ms := sw.elapsed().milliseconds()
		print('decode_ms=')
		println(ms)
		if ms < best {
			best = ms
		}
	}
	print('decode_best_ms=')
	println(best)
	st := pld.stats_snapshot()
	print('pld_drafts=')
	println(st.drafts)
	print('pld_accepted=')
	println(st.accepted)
	print('pld_accept_rate=')
	println(st.accept_rate())
	println('FORWARD-BENCH-OK')
}
