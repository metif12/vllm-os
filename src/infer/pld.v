module infer

// Prompt Lookup Decoding (PLD / ngram speculation), phase 8.
//
// At each greedy decode step, find the longest n-gram suffix of the token
// history (prompt + own output) that appeared earlier, and propose the
// following tokens as drafts. Each draft is verified by running the real
// forward and comparing the greedy argmax; a mismatch emits the model's
// token instead. Every emitted token is therefore the model's argmax at its
// position — output is BIT-IDENTICAL to plain greedy decoding on/off.
//
// This foundation is intentionally serial: verification runs one forward per
// draft, so it costs the same as plain decode (plus bookkeeping). The
// wall-time win needs a batched verifier (multi-token matmul + batched causal
// attention), which is a separate engine change — recorded in the roadmap.
// Pure V, no allocation in the token loop: the caller owns drafts/emitted.

pub struct PldCfg {
pub:
	enabled      bool
	max_ngram    int = 3
	continuation int = 10
}

pub struct PldStats {
pub mut:
	rounds   int
	drafts   int
	accepted int
}

pub struct PldController {
pub:
	cfg PldCfg
mut:
	history []int
	drafts  []int
	stats   PldStats
	// batch is the shared-weight verifier scratch (nil = serial verify).
	batch &BatchCtx = unsafe { nil }
}

// enable_batch prepares the shared-weight batched verifier for this model.
// Returns false (leaving the serial path active) for quant mixes the batched
// kernels do not cover.
pub fn (mut c PldController) enable_batch(cfg Qwen3Config, gguf GgufModel) bool {
	mut bc := new_batch_ctx(cfg, gguf, c.drafts.len) or { return false }
	c.batch = &bc
	return true
}

pub fn new_pld_controller(cfg PldCfg, prompt []int) PldController {
	mut cont := cfg.continuation
	if cont < 1 {
		cont = 1
	}
	return PldController{
		cfg:     cfg
		history: prompt.clone()
		drafts:  []int{len: cont}
	}
}

// pld_draft writes up to `continuation` followers of the longest matching
// n-gram suffix of history into out. Returns the number written (0 = no
// match). A match must not overlap the suffix being matched.
@[direct_array_access]
fn pld_draft(history []int, max_ngram int, continuation int, mut out []int) int {
	n := history.len
	if n < 2 || max_ngram < 1 || continuation < 1 {
		return 0
	}
	max_l := if max_ngram < n - 1 { max_ngram } else { n - 1 }
	for l := max_l; l >= 1; l-- {
		// Most recent match first, like PLD's lookup over prompt + output.
		for p := n - l - 1; p >= 0; p-- {
			if p + l > n - l {
				continue // would overlap the suffix
			}
			mut ok := true
			for k := 0; k < l; k++ {
				if history[p + k] != history[n - l + k] {
					ok = false
					break
				}
			}
			if !ok {
				continue
			}
			avail := n - (p + l)
			mut cnt := if avail < continuation { avail } else { continuation }
			if cnt > out.len {
				cnt = out.len
			}
			for k := 0; k < cnt; k++ {
				out[k] = history[p + l + k]
			}
			return cnt
		}
	}
	return 0
}

fn pld_advance(mut f ForwardCtx, blob []u8, gguf GgufModel, id int) ! {
	embed_gather(blob, gguf, f.cfg.hidden, id, mut f.hidden)!
	f.decode_step()!
}

// next_tokens produces the next 1..continuation+1 tokens into out and
// returns the count. With PLD disabled it is exactly one plain greedy step.
pub fn (mut c PldController) next_tokens(mut f ForwardCtx, blob []u8, gguf GgufModel, mut out []int) !int {
	c.stats.rounds++
	mut nd := 0
	if c.cfg.enabled {
		nd = pld_draft(c.history, c.cfg.max_ngram, c.cfg.continuation, mut c.drafts)
	}
	if nd == 0 {
		next := sample_greedy(f.logits)
		out[0] = next
		c.history << next
		pld_advance(mut f, blob, gguf, next)!
		return 1
	}

	c.stats.drafts += nd
	if c.batch != unsafe { nil } {
		return c.verify_batched(mut f, blob, gguf, nd, mut out)
	}
	mut written := 0
	for i := 0; i < nd; i++ {
		pred := sample_greedy(f.logits)
		if pred != c.drafts[i] {
			out[written] = pred
			written++
			c.history << pred
			pld_advance(mut f, blob, gguf, pred)!
			return written
		}
		c.stats.accepted++
		out[written] = c.drafts[i]
		written++
		c.history << c.drafts[i]
		pld_advance(mut f, blob, gguf, c.drafts[i])!
	}
	// All drafts accepted: the model's argmax after feeding the last draft is
	// the bonus token (standard speculative decoding).
	bonus := sample_greedy(f.logits)
	out[written] = bonus
	written++
	c.history << bonus
	pld_advance(mut f, blob, gguf, bonus)!
	return written
}

// verify_batched runs the whole draft block in one shared-weight batch,
// then compares the per-position argmaxes in order. On rejection it rolls
// the KV cache back to the accepted prefix and feeds the model's token.
fn (mut c PldController) verify_batched(mut f ForwardCtx, blob []u8, gguf GgufModel, nd int, mut out []int) !int {
	start_pos := f.pos
	vocab := f.logits.len
	// decode_batch overwrites f.logits with the last draft row, so snapshot
	// the authoritative distribution for the first position before it runs.
	mut first_logits := []f32{len: vocab}
	for j, v in f.logits {
		first_logits[j] = v
	}
	mut toks := []int{len: nd}
	for i := 0; i < nd; i++ {
		toks[i] = c.drafts[i]
	}
	f.decode_batch(mut c.batch, toks)!
	mut written := 0
	for i := 0; i < nd; i++ {
		// The first draft position is verified against the distribution we
		// already have (f.logits from the previous step) — not a recomputed
		// batch row — so the emitted token matches the serial path exactly.
		mut row := unsafe { first_logits[0..vocab] }
		if i > 0 {
			row = c.batch.logits[i * vocab..(i + 1) * vocab]
		}
		pred := sample_greedy(row)
		if pred != c.drafts[i] {
			// The batch appended all nd draft rows already. Truncate to the
			// accepted prefix, then feed the model's token from that prefix.
			for l in 0 .. f.cfg.layers {
				for kh in 0 .. f.cfg.heads_kv {
					f.k_cache[l * f.cfg.heads_kv + kh].truncate_to(start_pos + i)!
					f.v_cache[l * f.cfg.heads_kv + kh].truncate_to(start_pos + i)!
				}
			}
			f.pos = start_pos + i
			out[written] = pred
			written++
			c.history << pred
			pld_advance(mut f, blob, gguf, pred)!
			return written
		}
		c.stats.accepted++
		out[written] = c.drafts[i]
		written++
		c.history << c.drafts[i]
	}
	bonus := sample_greedy(f.logits)
	out[written] = bonus
	written++
	c.history << bonus
	pld_advance(mut f, blob, gguf, bonus)!
	return written
}

pub fn (c &PldController) stats_snapshot() PldStats {
	return c.stats
}

// accept_rate is accepted drafts / attempted drafts (0 when none attempted).
pub fn (s PldStats) accept_rate() f64 {
	if s.drafts == 0 {
		return 0
	}
	return f64(s.accepted) / f64(s.drafts)
}
