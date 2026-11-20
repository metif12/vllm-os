module main

// Phase 8 speculation gate (host-only, real GGUF): times greedy generate
// with PLD off (serial decode) vs PLD on (shared-weight batched verify),
// and reports draft accept rate. Same box, same prompt, back-to-back.
//
// Usage:
//   v -prod -cc gcc -path "src|@vlib|@vmodules" -o pldbench tools/pld_bench.v
//   ./pldbench models/Qwen3-0.6B-Q4_0.gguf [n_tokens]
// Optional VLLM_PROBE overrides the (input-grounded) code-edit probe.
// VLLM_PLD_BATCH=1 opts into the EXPERIMENTAL shared-weight batch verifier;
// the default is bit-exact serial verification (see ROADMAP phase 8).
//
// NOTE: print/println only, no string interpolation (release-V -prod cgen
// quirk in tiny module-main binaries).
import infer
import os
import time

const probe = 'def add(a, b):\n    result = a + b\n    return result\n\n# add two numbers\ndef sum(x, y):\n    result = x + y\n    return result\n\n# add three numbers\ndef total(a, b, c):\n    result = '

fn main() {
	weights := os.args[1]
	n := if os.args.len > 2 { os.args[2].int() } else { 16 }
	mut m := infer.load_model(weights, infer.Qwen3Config{}) or { panic(err) }
	if m.blob.len == 0 {
		panic('pld_bench needs a cached GGUF blob')
	}
	sp := infer.SampleParams{
		temp: 0.0
	}
	mut text := os.getenv('VLLM_PROBE')
	if text == '' {
		text = probe
	}
	mut ngram := 3
	if v := os.getenv_opt('VLLM_PLD_NGRAM') {
		ngram = v.int()
	}
	mut cont := 10
	if v := os.getenv_opt('VLLM_PLD_CONTINUATION') {
		cont = v.int()
	}
	// Warm up both paths once (page cache + code paths).
	m.pld = infer.PldCfg{
		enabled: false
	}
	_ = m.generate(text, 2, sp)
	m.pld = infer.PldCfg{
		enabled:      true
		max_ngram:    3
		continuation: 10
	}
	_ = m.generate(text, 2, sp)

	m.pld = infer.PldCfg{
		enabled: false
	}
	mut sw := time.new_stopwatch()
	plain := m.generate(text, n, sp)
	plain_ms := sw.elapsed().milliseconds()

	m.pld = infer.PldCfg{
		enabled:      os.getenv('VLLM_PLD_OFF') == ''
		max_ngram:    ngram
		continuation: cont
	}
	mut stats := infer.PldStats{}
	m.pld_last = &stats
	sw = time.new_stopwatch()
	spec := m.generate(text, n, sp)
	spec_ms := sw.elapsed().milliseconds()

	print('tokens=')
	println(n.str())
	print('plain_ms=')
	println(plain_ms.str())
	print('pld_ms=')
	println(spec_ms.str())
	print('same_text=')
	println((plain == spec).str())
	print('ngram=')
	println(ngram.str())
	print('continuation=')
	println(cont.str())
	print('drafts=')
	println(stats.drafts.str())
	print('accepted=')
	println(stats.accepted.str())
	print('accept_rate=')
	println(stats.accept_rate().str())
	print('plain_head=')
	println(plain)
	print('pld_head=')
	println(spec)
	println('PLD-BENCH-OK')
}
