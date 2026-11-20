module infer

// Single-entry prompt-prefix KV cache (phase 9, host-only, opt-in).
//
// Multi-turn chat re-sends the whole history every request, so the shared
// prefix is re-prefilled on every turn. This cache keeps the KV state of the
// most recent prompt; when a later request starts with those exact token
// ids, its KV rows are copied into the fresh ForwardCtx and only the new
// suffix is prefilled. One entry bounds memory to one prompt's KV.
//
// The cache is shared across request handler threads, so every access is
// under a Mutex. It is off by default (config `prefix_cache`); adding shared
// mutable state to the serve path is an explicit opt-in decision.
import sync

pub struct PrefixCache {
mut:
	mutex   &sync.Mutex = sync.new_mutex()
	tokens  []int
	pos     int
	k_cache []KVCache
	v_cache []KVCache
	hits    int
	misses  int
}

pub fn new_prefix_cache() &PrefixCache {
	return &PrefixCache{}
}

// match_len returns how many leading ids match the cached prompt. It never
// returns the full request (there must be at least one token to prefill).
pub fn (mut c PrefixCache) match_len(ids []int) int {
	c.mutex.lock()
	defer {
		c.mutex.unlock()
	}
	n := c.tokens.len
	if n == 0 || n >= ids.len {
		return 0
	}
	for i in 0 .. n {
		if c.tokens[i] != ids[i] {
			return 0
		}
	}
	return n
}

// copy_into copies the cached KV prefix into a fresh ForwardCtx and advances
// its position. The caller prefills only ids[n..].
pub fn (mut c PrefixCache) copy_into(mut f ForwardCtx, n int) ! {
	c.mutex.lock()
	defer {
		c.mutex.unlock()
	}
	if n < 1 || n > c.pos || c.k_cache.len != f.k_cache.len || c.v_cache.len != f.v_cache.len {
		return error('prefix_cache: bad copy n=${n} cached=${c.pos}')
	}
	for l in 0 .. c.k_cache.len {
		src := c.k_cache[l]
		if src.dim != f.k_cache[l].dim {
			return error('prefix_cache: dim mismatch at layer ${l}')
		}
		mut dst := &f.k_cache[l]
		for i in 0 .. n * src.dim {
			dst.data[i] = src.data[i]
		}
		dst.len = n
		sv := c.v_cache[l]
		if sv.dim != f.v_cache[l].dim {
			return error('prefix_cache: vdim mismatch at layer ${l}')
		}
		mut dv := &f.v_cache[l]
		for i in 0 .. n * sv.dim {
			dv.data[i] = sv.data[i]
		}
		dv.len = n
	}
	f.pos = n
	c.hits++
}

// store replaces the cache with the first n positions of f's KV state.
pub fn (mut c PrefixCache) store(ids []int, n int, f &ForwardCtx) ! {
	if n < 1 || n > ids.len {
		return error('prefix_cache: bad store n=${n} ids=${ids.len}')
	}
	mut kc := []KVCache{cap: f.k_cache.len}
	mut vc := []KVCache{cap: f.v_cache.len}
	for src in f.k_cache {
		mut d := new_kv_cache(src.dim, src.cap)!
		if n > src.cap {
			return error('prefix_cache: n exceeds cache cap')
		}
		for i in 0 .. n * src.dim {
			d.data[i] = src.data[i]
		}
		d.len = n
		kc << d
	}
	for src in f.v_cache {
		mut d := new_kv_cache(src.dim, src.cap)!
		for i in 0 .. n * src.dim {
			d.data[i] = src.data[i]
		}
		d.len = n
		vc << d
	}
	c.mutex.lock()
	defer {
		c.mutex.unlock()
	}
	c.tokens = ids[0..n].clone()
	c.pos = n
	c.k_cache = kc
	c.v_cache = vc
	c.misses++
}

pub fn (c &PrefixCache) stats() (int, int) {
	return c.hits, c.misses
}
