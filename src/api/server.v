module api

import cfg
import infer
import net
import strconv
import strings
import time

// Raw-socket HTTP server on purpose: `veb` pulls in OS deps we cannot
// satisfy on vinix-mini. This stays portable to bare metal. Routing lives
// in routes.v, handlers in handlers.v — this file owns only the socket
// loop, request framing, and response bytes.

pub struct Server {
pub:
	port int = 8000
mut:
	model &infer.Model
}

// serve starts the OpenAI-compatible HTTP API. Limits (max_conns,
// max_body) come from cc so deployments tune backpressure without
// code changes.
pub fn serve(cc cfg.Config, model &infer.Model) ! {
	ctx := &HandlerCtx{
		model:  model
		config: cc
	}
	mut ln := net.listen_tcp(.ip6, ':${cc.server.port}')!
	eprintln('[vllm-os] openai-compat api on :${cc.server.port} model=${model.cfg.model_id} quant=${model.cfg.quant}')
	mut sema := chan int{cap: cc.server.max_conns}
	// The auth/rate gate runs here, in the single accept-loop thread,
	// before any spawn: rejected junk never costs a thread, and the
	// bucket map needs no locks. See limits.v for why ctx stays immutable.
	mut buckets := map[string]Bucket{}
	for {
		mut conn := ln.accept()!
		head, body := read_request(mut conn, cc.server.max_body) or {
			conn.close() or {}
			continue
		}
		parts := head.split_into_lines()
		if parts.len == 0 {
			conn.close() or {}
			continue
		}
		method_path := parts[0].split(' ')
		if method_path.len < 2 {
			conn.close() or {}
			continue
		}
		// /health stays open for load balancers; everything else needs
		// the bearer token when one is configured (empty = open mode).
		is_health := method_path[0] == 'GET' && method_path[1] == '/health'
		if !is_health && !authorized(head, cc.server.auth_token) {
			send_response(mut conn, 401, 'application/json', '{"error":"unauthorized"}')
			conn.close() or {}
			continue
		}
		if cc.server.rate_per_sec > 0 {
			ip := conn.peer_ip() or { '' }
			if !allow(mut buckets, ip, time.now().unix(), cc.server.rate_per_sec,
				cc.server.rate_burst) {
				send_response(mut conn, 429, 'application/json', '{"error":"rate limited"}')
				conn.close() or {}
				continue
			}
		}
		sema <- 1
		spawn handle(mut conn, ctx, sema, head, body)
	}
}

fn sema_release(s chan int) {
	token := <-s
	_ = token
}

// read_request reads headers + body bytes, capped at max_body (configured;
// rejects absurd lengths before alloc). Bodies arrive either Content-Length
// framed or Transfer-Encoding: chunked (uclient-fetch and friends send
// small POSTs chunked with no Content-Length at all).
fn read_request(mut conn net.TcpConn, max_body int) !(string, string) {
	mut acc := []u8{cap: 8192}
	mut tmp := []u8{len: 4096}
	mut head_end := -1
	for head_end < 0 {
		n := conn.read(mut tmp) or { return error('closed') }
		if n <= 0 {
			return error('closed')
		}
		acc << tmp[..n]
		if acc.len > 16384 + max_body {
			return error('request too large')
		}
		s := acc.bytestr()
		if idx := s.index('\r\n\r\n') {
			head_end = idx
		}
	}
	head := acc[..head_end].bytestr()
	if has_chunked(head) {
		body := read_chunked_body(mut acc, head_end + 4, mut conn, max_body)!
		return head, body
	}
	need := content_length(head)
	if need > max_body {
		return error('body too large')
	}
	total_needed := head_end + 4 + need
	for acc.len < total_needed {
		n := conn.read(mut tmp) or { return error('closed') }
		if n <= 0 {
			return error('closed')
		}
		acc << tmp[..n]
		if acc.len > 16384 + max_body {
			return error('request too large')
		}
	}
	full := acc.bytestr()
	return full[..head_end], full[head_end + 4..total_needed]
}

fn content_length(head string) int {
	for line in head.split_into_lines() {
		if line.len > 15 && line[..15].to_lower() == 'content-length:' {
			return line[15..].trim_space().int()
		}
	}
	return 0
}

// has_chunked reports whether the header block selects Transfer-Encoding:
// chunked (matched case-insensitively; other codings alongside are fine).
fn has_chunked(head string) bool {
	for line in head.split_into_lines() {
		if line.len > 18 && line[..18].to_lower() == 'transfer-encoding:' {
			if line[18..].to_lower().contains('chunked') {
				return true
			}
		}
	}
	return false
}

// read_chunked_body de-chunks the body starting at pos (just past the
// header terminator), topping up acc from conn as needed. Chunk extensions
// and trailers are skipped; the total is capped at max_body like the
// Content-Length path.
fn read_chunked_body(mut acc []u8, pos int, mut conn net.TcpConn, max_body int) !string {
	mut out := []u8{cap: 4096}
	mut tmp := []u8{len: 4096}
	mut at := pos
	for {
		mut eol := -1
		for {
			if idx := acc.bytestr().index_after('\r\n', at) {
				eol = idx
				break
			}
			n := conn.read(mut tmp) or { return error('closed') }
			if n <= 0 {
				return error('closed')
			}
			acc << tmp[..n]
			if acc.len > 16384 + max_body {
				return error('request too large')
			}
		}
		line := acc[at..eol].bytestr()
		size_str := if i := line.index(';') { line[..i] } else { line }
		size64 := strconv.parse_int(size_str.trim_space(), 16, 32) or {
			return error('bad chunk size')
		}
		if size64 < 0 || size64 > i64(max_body) {
			return error('bad chunk size')
		}
		size := int(size64)
		at = eol + 2
		if size == 0 {
			// Consume trailers + final CRLF (usually just one empty line).
			for {
				if idx := acc.bytestr().index_after('\r\n', at) {
					if idx == at {
						return out.bytestr()
					}
					at = idx + 2
					continue
				}
				n := conn.read(mut tmp) or { return error('closed') }
				if n <= 0 {
					return error('closed')
				}
				acc << tmp[..n]
				if acc.len > 16384 + max_body {
					return error('request too large')
				}
			}
		}
		for acc.len < at + size + 2 {
			n := conn.read(mut tmp) or { return error('closed') }
			if n <= 0 {
				return error('closed')
			}
			acc << tmp[..n]
			if acc.len > 16384 + max_body {
				return error('request too large')
			}
		}
		out << acc[at..at + size]
		if out.len > max_body {
			return error('body too large')
		}
		at += size + 2
	}
	return out.bytestr()
}

fn handle(mut conn net.TcpConn, ctx &HandlerCtx, sema chan int, head string, body string) {
	defer {
		sema_release(sema)
		conn.close() or {}
	}
	lines := head.split_into_lines()
	if lines.len == 0 {
		return
	}
	parts := lines[0].split(' ')
	if parts.len < 2 {
		return
	}
	route := match_route(parts[0], parts[1]) or {
		send_response(mut conn, 404, 'application/json',
			'{"error":"not found, try /health or /v1/models"}')
		return
	}
	handler := route.handler or {
		send_response(mut conn, 500, 'application/json', '{"error":"unwired route"}')
		return
	}

	res := handler(body, ctx)
	send_response(mut conn, res.status, res.content_type, res.body)
}

fn send_response(mut conn net.TcpConn, code int, content_type string, payload string) {
	status := if code == 200 {
		'OK'
	} else if code == 400 {
		'Bad Request'
	} else if code == 401 {
		'Unauthorized'
	} else if code == 429 {
		'Too Many Requests'
	} else {
		'Not Found'
	}
	mut sb := strings.new_builder(160 + payload.len)
	sb.write_string('HTTP/1.1 ')
	sb.write_string(code.str())
	sb.write_string(' ')
	sb.write_string(status)
	sb.write_string('\r\nContent-Type: ')
	sb.write_string(content_type)
	sb.write_string('\r\nContent-Length: ')
	sb.write_string(payload.len.str())
	sb.write_string('\r\nConnection: close\r\n\r\n')
	sb.write_string(payload)
	conn.write_string(sb.str()) or {}
}

// escape_json escapes one pass (old code chained 3 full-string replaces).
pub fn escape_json(s string) string {
	hex_digits := '0123456789abcdef'
	mut sb := strings.new_builder(s.len + 16)
	for ch in s {
		match ch {
			`\\` {
				sb.write_string('\\\\')
			}
			`"` {
				sb.write_string('\\"')
			}
			`\n` {
				sb.write_string('\\n')
			}
			`\r` {
				sb.write_string('\\r')
			}
			`\t` {
				sb.write_string('\\t')
			}
			else {
				if ch < 0x20 {
					sb.write_string('\\u00')
					sb.write_u8(hex_digits[ch >> 4])
					sb.write_u8(hex_digits[ch & 0xf])
				} else {
					sb.write_u8(ch)
				}
			}
		}
	}
	return sb.str()
}
