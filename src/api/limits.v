module api

// Request gate: bearer auth + per-IP token-bucket rate limiting.
//
// DESIGN NOTE (pinned V 0.5.2): `spawn f(mut x)` in the accept loop is
// unsound here — passing the shared ctx mutably crashed handler threads
// with garbage string headers (V panic: negative .cap in escape_json,
// bisected 2026-11-18: immutable spawn = green). So the gate runs in the
// SINGLE accept-loop thread before spawning: ctx stays immutable, the
// bucket map is main-thread-only, and no mutex exists anywhere in api/.
// The price (framing requests serially before spawn) is negligible next
// to model decode.

// authorized reports whether the request head carries the expected
// Bearer token. Empty expected token = open mode (no header required).
// /health stays open regardless — the caller (server.v) decides that.
pub fn authorized(head string, expected string) bool {
	if expected == '' {
		return true
	}
	for line in head.split_into_lines() {
		if line.len > 14 && line[..14].to_lower() == 'authorization:' {
			cred := line[14..].trim_space()
			if cred.len > 7 && cred[..7].to_lower() == 'bearer ' {
				return cred[7..] == expected
			}
			return false
		}
	}
	return false
}

// Bucket is one token-bucket: tokens refill at per_sec, capped at the
// burst allowance (or per_sec when burst is 0).
pub struct Bucket {
pub mut:
	tokens  f64
	updated i64
}

// allow_at consumes one token at time now (unix seconds). Pure: the
// caller owns storage, which keeps unit tests clock-free.
pub fn (mut b Bucket) allow_at(now i64, per_sec int, burst int) bool {
	eff := if burst > 0 { burst } else { per_sec }
	b.tokens += f64(now - b.updated) * f64(per_sec)
	if b.tokens > f64(eff) {
		b.tokens = f64(eff)
	}
	b.updated = now
	if b.tokens >= 1.0 {
		b.tokens -= 1.0
		return true
	}
	return false
}

// allow is the shared per-IP gate: true = request may proceed. Unknown
// peers start with a full bucket. Main-thread-only storage by design
// (see note above) — callers pass their own map, tests use a local one.
pub fn allow(mut buckets map[string]Bucket, key string, now i64, per_sec int, burst int) bool {
	eff := if burst > 0 { burst } else { per_sec }
	mut b := buckets[key] or {
		Bucket{
			tokens:  f64(eff)
			updated: now
		}
	}
	ok := b.allow_at(now, per_sec, burst)
	buckets[key] = b
	return ok
}
