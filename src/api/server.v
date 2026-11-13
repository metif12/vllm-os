module api

import cfg
import infer
import net
import strings

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
	for {
		mut conn := ln.accept()!
		sema <- 1
		spawn handle(mut conn, ctx, sema)
	}
}

fn sema_release(s chan int) {
	token := <-s
	_ = token
}

// read_request reads headers + exactly Content-Length body bytes, capped
// at max_body (configured; rejects absurd Content-Length before alloc).
fn read_request(mut conn net.TcpConn, max_body int) !(string, string) {
	mut acc := []u8{cap: 8192}
	mut tmp := []u8{len: 4096}
	mut head_end := -1
	mut total_needed := -1
	for {
		n := conn.read(mut tmp) or { return error('closed') }
		if n <= 0 {
			return error('closed')
		}
		acc << tmp[..n]
		if acc.len > 16384 + max_body {
			return error('request too large')
		}
		if head_end < 0 {
			s := acc.bytestr()
			if idx := s.index('\r\n\r\n') {
				head_end = idx
				total_needed = head_end + 4 + content_length(s[..head_end])
				if total_needed - (head_end + 4) > max_body {
					return error('body too large')
				}
			}
		}
		if head_end >= 0 && acc.len >= total_needed {
			break
		}
	}
	full := acc.bytestr()
	return full[..head_end], full[head_end + 4..total_needed]
}

fn content_length(head string) int {
	for line in head.split_into_lines() {
		if line.len > 15 && (line[..15].to_lower() == 'content-length:') {
			return line[15..].trim_space().int()
		}
	}
	return 0
}

fn handle(mut conn net.TcpConn, ctx &HandlerCtx, sema chan int) {
	defer {
		sema_release(sema)
		conn.close() or {}
	}
	head, body := read_request(mut conn, ctx.config.server.max_body) or { return }
	lines := head.split_into_lines()
	if lines.len == 0 {
		return
	}
	parts := lines[0].split(' ')
	if parts.len < 2 {
		return
	}
	route := match_route(parts[0], parts[1]) or {
		send_response(mut conn, 404, 'application/json', '{"error":"not found, try /health or /v1/models"}')
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
			`\\` { sb.write_string('\\\\') }
			`"` { sb.write_string('\\"') }
			`\n` { sb.write_string('\\n') }
			`\r` { sb.write_string('\\r') }
			`\t` { sb.write_string('\\t') }
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
