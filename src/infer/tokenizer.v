module infer

// Tokenizer is the extension point for text encoding. The generate path
// calls only encode, so a new tokenizer (byte-level, test fake, future
// real BPE merges) implements this interface and plugs into Model with
// no other changes. Bpe degrades to stub hashing when no tokenizer.json
// was baked in; StubTokenizer is the explicit no-weights variant.
pub interface Tokenizer {
	// encode maps text to token ids.
	encode(text string) []int
	// decode maps token ids back to text (inverse byte mapping).
	decode(ids []int) string
	// name reports the tokenizer variant for boot logs.
	name() string
}

// BPE tokenizer: loads HF tokenizer.json (Qwen3) when baked into the image,
// falls back to the whitespace stub when absent (keeps the HTTP surface testable).
pub struct Bpe {
pub:
	vocab  map[string]int
	merges map[string]int
	rev    map[int]string // inverse vocab for decode
	bpe_ok bool
}

// name reports 'bpe' when merges loaded, 'bpe-stub' on the hash fallback.
pub fn (b Bpe) name() string {
	if b.bpe_ok {
		return 'bpe'
	}
	return 'bpe-stub'
}

// StubTokenizer always hashes (no-weights dev loop, tests).
pub struct StubTokenizer {}

// name reports the stub variant.
pub fn (t StubTokenizer) name() string {
	return 'stub'
}

// encode hashes whitespace-separated words (see encode_stub).
pub fn (t StubTokenizer) encode(text string) []int {
	return encode_stub(text)
}

// decode renders ids as placeholders (no-weights dev loop).
pub fn (t StubTokenizer) decode(ids []int) string {
	return decode_stub(ids)
}

pub struct ChatMsg {
pub:
	role    string
	content string
}

// encode_stub hashes whitespace-separated words (no-weights fallback).
pub fn encode_stub(text string) []int {
	parts := text.split(' ')
	mut out := []int{cap: parts.len}
	for p in parts {
		if p.len == 0 {
			continue
		}
		out << p.len * 131 + p[0]
	}
	return out
}

// decode_stub renders token ids as <tok_N> placeholders (stub fallback).
pub fn decode_stub(ids []int) string {
	return ids.map('<tok_${it}>').join(' ')
}

// decode renders ids to text: inverse vocab lookup, then reverse the
// ByteLevel mapping (each piece char back to its byte), yielding the
// original UTF-8. Unknown ids degrade to the stub placeholder.
pub fn (b Bpe) decode(ids []int) string {
	if !b.bpe_ok {
		return decode_stub(ids)
	}
	mut raw := []u8{cap: ids.len * 8}
	for id in ids {
		if s := b.rev[id] {
			for i := 0; i < s.len; {
				r, w := bpe_rune_at(s, i)
				raw << bpe_unicode_to_byte(r)
				i += w
			}
		} else {
			for c in '<tok_${id}>'.bytes() {
				raw << c
			}
		}
	}
	return raw.bytestr()
}

// bpe_unicode_to_byte inverts bpe_byte_unicode (piece char -> raw byte).
fn bpe_unicode_to_byte(r u32) u8 {
	if r <= 0x7e {
		return u8(r)
	}
	if r >= 0xa1 && r <= 0xff {
		return u8(r)
	}
	// U+0100 and up: rank determines the byte (mirror of byte_unicode).
	n := int(r) - 0x100
	if n <= 32 {
		return u8(n)
	}
	if n == 33 {
		return u8(0x7f)
	}
	if n <= 66 {
		return u8(0x80 + n - 34)
	}
	return u8(0xad)
}

// load_bpe parses a HF tokenizer.json (model.vocab + model.merges).
// Merge pairs are keyed as "left + ' ' + right" (raw space never appears
// inside a ByteLevel piece, so the join is unambiguous).
//
// NOTE: this uses a purpose-built scanner, not json.decode: V's generic
// decoder takes ~3.5 minutes on this 11 MB / 300k-entry file (measured),
// while the scanner runs in ~2s. Only the model.vocab object and the
// model.merges array are parsed; the rest of the document is skipped.
// Bare-metal note: file IO stays host-side (the kernel port bakes the
// parsed tables instead).
pub fn load_bpe(path string) !Bpe {
	raw := read_whole_file(path)!
	mut p := BpeJsonParser{
		s: raw.bytestr()
	}
	return p.parse()!
}

struct BpeJsonParser {
	s string
mut:
	pos int
}

fn (mut p BpeJsonParser) parse() !Bpe {
	// Locate the model.vocab object and model.merges array by key search
	// (whitespace-tolerant: HF pretty-prints with spaces after colons).
	vi := p.s.index('"vocab"') or { return error('bpe: no "vocab" in tokenizer.json') }
	p.pos = vi + '"vocab"'.len
	p.skip_ws()
	p.expect(`:`)!
	p.skip_ws()
	p.expect(`{`)!
	mut vocab := map[string]int{}
	p.parse_vocab(mut vocab)!
	mut mi := p.s.index_after('"merges"', 0) or {
		return error('bpe: no "merges" in tokenizer.json')
	}
	// merges must come after vocab in a HF file; if not, rescan from vocab end.
	if mi < p.pos {
		mi = p.s.index_after('"merges"', p.pos) or {
			return error('bpe: no "merges" after vocab in tokenizer.json')
		}
	}
	p.pos = mi + '"merges"'.len
	p.skip_ws()
	p.expect(`:`)!
	p.skip_ws()
	p.expect(`[`)!
	mut merges := map[string]int{}
	p.parse_merges(mut merges)!
	mut rev := map[int]string{}
	for s, id in vocab {
		rev[id] = s
	}
	return Bpe{
		vocab:  vocab
		merges: merges
		rev:    rev
		bpe_ok: true
	}
}

// parse_vocab parses "k":v pairs up to the closing brace.
fn (mut p BpeJsonParser) parse_vocab(mut vocab map[string]int) ! {
	p.skip_ws()
	if p.peek() == `}` {
		p.pos++
		return
	}
	for {
		p.skip_ws()
		key := p.parse_string()!
		p.skip_ws()
		p.expect(`:`)!
		p.skip_ws()
		val := p.parse_int()!
		vocab[key] = val
		p.skip_ws()
		c := p.peek()
		if c == `,` {
			p.pos++
			continue
		}
		if c == `}` {
			p.pos++
			return
		}
		return error('bpe: expected , or } in vocab')
	}
}

// parse_merges parses ["a","b"] pairs up to the closing bracket.
fn (mut p BpeJsonParser) parse_merges(mut merges map[string]int) ! {
	p.skip_ws()
	if p.peek() == `]` {
		p.pos++
		return
	}
	mut rank := 0
	for {
		p.skip_ws()
		p.expect(`[`)!
		p.skip_ws()
		a := p.parse_string()!
		p.skip_ws()
		p.expect(`,`)!
		p.skip_ws()
		b := p.parse_string()!
		p.skip_ws()
		p.expect(`]`)!
		merges[a + ' ' + b] = rank
		rank++
		p.skip_ws()
		c := p.peek()
		if c == `,` {
			p.pos++
			continue
		}
		if c == `]` {
			p.pos++
			return
		}
		return error('bpe: expected , or ] in merges')
	}
}

fn (mut p BpeJsonParser) peek() u8 {
	if p.pos >= p.s.len {
		return 0
	}
	return p.s[p.pos]
}

fn (mut p BpeJsonParser) expect(c u8) ! {
	if p.peek() != c {
		return error('bpe: expected `${c.ascii_str()}` at offset ${p.pos}')
	}
	p.pos++
}

fn (mut p BpeJsonParser) skip_ws() {
	for p.pos < p.s.len {
		c := p.s[p.pos]
		if c == ` ` || c == `\t` || c == `\n` || c == `\r` {
			p.pos++
		} else {
			break
		}
	}
}

// parse_int parses an optional-minus digit run.
fn (mut p BpeJsonParser) parse_int() !int {
	mut neg := false
	if p.peek() == `-` {
		neg = true
		p.pos++
	}
	mut v := 0
	mut digits := 0
	for p.pos < p.s.len {
		c := p.s[p.pos]
		if c < `0` || c > `9` {
			break
		}
		v = v * 10 + int(c - `0`)
		p.pos++
		digits++
	}
	if digits == 0 {
		return error('bpe: expected integer at offset ${p.pos}')
	}
	if neg {
		v = -v
	}
	return v
}

// parse_string parses a JSON string with escapes (\" \\ \/ \b \f \n \r \t
// \uXXXX incl. surrogate pairs) into a V UTF-8 string.
fn (mut p BpeJsonParser) parse_string() !string {
	p.expect(`"`)!
	mut out := []u8{cap: 32}
	for {
		if p.pos >= p.s.len {
			return error('bpe: unterminated string')
		}
		c := p.s[p.pos]
		if c == `"` {
			p.pos++
			return out.bytestr()
		}
		if c != `\\` {
			out << c
			p.pos++
			continue
		}
		p.pos++
		if p.pos >= p.s.len {
			return error('bpe: dangling backslash')
		}
		e := p.s[p.pos]
		match e {
			`"` {
				out << u8(`"`)
			}
			`\\` {
				out << u8(`\\`)
			}
			`/` {
				out << u8(`/`)
			}
			`b` {
				out << u8(0x08)
			}
			`f` {
				out << u8(0x0c)
			}
			`n` {
				out << u8(`\n`)
			}
			`r` {
				out << u8(`\r`)
			}
			`t` {
				out << u8(`\t`)
			}
			`u` {
				p.pos++
				cp := p.parse_hex4()!
				if cp >= 0xd800 && cp <= 0xdbff {
					// high surrogate: expect \uDC00-\uDFFF
					if p.pos + 5 >= p.s.len || p.s[p.pos] != `\\` || p.s[p.pos + 1] != `u` {
						return error('bpe: lone high surrogate')
					}
					p.pos += 2
					lo := p.parse_hex4()!
					if lo < 0xdc00 || lo > 0xdfff {
						return error('bpe: bad low surrogate')
					}
					full := u32(0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00))
					p.append_utf8(mut out, full)
				} else {
					p.append_utf8(mut out, cp)
				}
				continue
			}
			else {
				return error('bpe: bad escape')
			}
		}

		p.pos++
	}
	return error('bpe: unterminated string')
}

// parse_hex4 parses exactly 4 hex digits into a code point.
fn (mut p BpeJsonParser) parse_hex4() !u32 {
	if p.pos + 4 > p.s.len {
		return error('bpe: truncated \\u escape')
	}
	mut v := u32(0)
	for i in 0 .. 4 {
		c := p.s[p.pos + i]
		d := if c >= `0` && c <= `9` {
			u32(c - `0`)
		} else if c >= `a` && c <= `f` {
			u32(c - `a` + 10)
		} else if c >= `A` && c <= `F` {
			u32(c - `A` + 10)
		} else {
			return error('bpe: bad hex digit in \\u escape')
		}
		v = v * 16 + d
	}
	p.pos += 4
	return v
}

// append_utf8 encodes a code point as UTF-8 bytes.
fn (mut p BpeJsonParser) append_utf8(mut out []u8, cp u32) {
	if cp < 0x80 {
		out << u8(cp)
	} else if cp < 0x800 {
		out << u8(0xc0 | (cp >> 6))
		out << u8(0x80 | (cp & 0x3f))
	} else if cp < 0x10000 {
		out << u8(0xe0 | (cp >> 12))
		out << u8(0x80 | ((cp >> 6) & 0x3f))
		out << u8(0x80 | (cp & 0x3f))
	} else {
		out << u8(0xf0 | (cp >> 18))
		out << u8(0x80 | ((cp >> 12) & 0x3f))
		out << u8(0x80 | ((cp >> 6) & 0x3f))
		out << u8(0x80 | (cp & 0x3f))
	}
}

// encode runs byte-level BPE: pre-tokenize (Qwen2 pattern), map each
// piece through the GPT-2 byte alphabet, apply merges by rank, look up
// vocab. Unknown bytes degrade to the stub hash (keeps the HTTP surface
// testable when tokenizer.json lacks a piece).
pub fn (b Bpe) encode(text string) []int {
	if !b.bpe_ok {
		return encode_stub(text)
	}
	mut out := []int{}
	for piece in bpe_pretokenize(text) {
		// ByteLevel: map each byte through the GPT-2 alphabet.
		mut parts := []string{cap: piece.len * 2}
		for i in 0 .. piece.len {
			parts << bpe_byte_unicode(piece[i])
		}
		// Merge loop: repeatedly merge the lowest-rank adjacent pair.
		for parts.len > 1 {
			mut best_rank := 1 << 30
			mut best_idx := -1
			for i in 0 .. parts.len - 1 {
				if r := b.merges[parts[i] + ' ' + parts[i + 1]] {
					if r < best_rank {
						best_rank = r
						best_idx = i
					}
				}
			}
			if best_idx < 0 {
				break
			}
			mut next_parts := []string{cap: parts.len - 1}
			for i in 0 .. parts.len {
				if i == best_idx {
					next_parts << parts[i] + parts[i + 1]
				} else if i == best_idx + 1 {
					continue
				} else {
					next_parts << parts[i]
				}
			}
			parts = next_parts.clone()
		}
		for p in parts {
			if id := b.vocab[p] {
				out << id
			} else {
				out << p.len * 131 + int(p[0])
			}
		}
	}
	return out
}

// bpe_byte_unicode maps one byte through the GPT-2 ByteLevel alphabet.
// ASCII prints as itself; bytes 0xA1-0xFF map to code points U+00A1-U+00FF
// (emitted as UTF-8, matching the tokenizer.json vocab keys); the rest
// shift to U+0100 and up (avoids whitespace/control collisions).
fn bpe_byte_unicode(b u8) string {
	if b >= 0x21 && b <= 0x7e {
		return b.ascii_str()
	}
	if (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff) {
		return [u8(0xc0 | (b >> 6)), u8(0x80 | (b & 0x3f))].bytestr()
	}
	// U+0100 + rank among the shifted bytes: 0x00-0x20 -> 0-32,
	// 0x7F -> 33, 0x80-0xA0 -> 34-66, 0xAD -> 67.
	mut n := 0
	if b <= 0x20 {
		n = int(b)
	} else if b == 0x7f {
		n = 33
	} else if b <= 0xa0 {
		n = 34 + int(b - 0x80)
	} else {
		n = 67 // 0xAD
	}
	// Encode U+0100+n as UTF-8 (always 2 bytes: 0xC4/0xC5 + low).
	cp := 0x100 + n
	return [u8(0xc0 | u8(cp >> 6)), u8(0x80 | u8(cp & 0x3f))].bytestr()
}

// Qwen3 chat template (ChatML-style). Applied server-side.
// apply_chat_template wraps messages with <|im_start|> frames.
pub fn apply_chat_template(messages []ChatMsg) string {
	mut s := ''
	for m in messages {
		s += '<|im_start|>${m.role}\n${m.content}<|im_end|>\n'
	}
	s += '<|im_start|>assistant\n'
	return s
}

// bpe_rune_at decodes one UTF-8 rune at byte offset (invalid sequences
// yield U+FFFD and advance 1, mirroring HF's lossless fallback).
fn bpe_rune_at(s string, off int) (u32, int) {
	b0 := s[off]
	if b0 < 0x80 {
		return u32(b0), 1
	}
	if b0 & 0xe0 == 0xc0 {
		if off + 1 < s.len {
			b1 := s[off + 1]
			if b1 & 0xc0 == 0x80 {
				r := (u32(b0 & 0x1f) << 6) | u32(b1 & 0x3f)
				if r >= 0x80 {
					return r, 2
				}
			}
		}
		return 0xfffd, 1
	}
	if b0 & 0xf0 == 0xe0 {
		if off + 2 < s.len {
			b1 := s[off + 1]
			b2 := s[off + 2]
			if b1 & 0xc0 == 0x80 && b2 & 0xc0 == 0x80 {
				r := (u32(b0 & 0x0f) << 12) | (u32(b1 & 0x3f) << 6) | u32(b2 & 0x3f)
				if r >= 0x800 && !(r >= 0xd800 && r <= 0xdfff) {
					return r, 3
				}
			}
		}
		return 0xfffd, 1
	}
	if b0 & 0xf8 == 0xf0 {
		if off + 3 < s.len {
			b1 := s[off + 1]
			b2 := s[off + 2]
			b3 := s[off + 3]
			if b1 & 0xc0 == 0x80 && b2 & 0xc0 == 0x80 && b3 & 0xc0 == 0x80 {
				r := (u32(b0 & 0x07) << 18) | (u32(b1 & 0x3f) << 12) | (u32(b2 & 0x3f) << 6) | u32(b3 & 0x3f)
				if r >= 0x10000 && r <= 0x10ffff {
					return r, 4
				}
			}
		}
		return 0xfffd, 1
	}
	return 0xfffd, 1
}

// bpe_letter_ranges covers \p{L} for the scripts real text uses (ASCII,
// Latin/Greek/Cyrillic/Armenian/Hebrew/Arabic, Indic, Thai/Lao/Tibetan/
// Myanmar/Georgian, CJK all planes, Kana, Hangul, fullwidth). Sorted for
// binary search. Ultra-rare scripts degrade per-rune (documented).
const bpe_letter_ranges = [
	[0x41, 0x5a],
	[0x61, 0x7a],
	[0xaa, 0xaa],
	[0xb5, 0xb5],
	[0xba, 0xba],
	[0xc0, 0xd6],
	[0xd8, 0xf6],
	[0xf8, 0x2c1],
	[0x2c6, 0x2d1],
	[0x2e0, 0x2e4],
	[0x2ec, 0x2ec],
	[0x2ee, 0x2ee],
	[0x370, 0x377],
	[0x37a, 0x37d],
	[0x37f, 0x37f],
	[0x386, 0x386],
	[0x388, 0x38a],
	[0x38c, 0x38c],
	[0x38e, 0x3a1],
	[0x3a3, 0x3f5],
	[0x3f7, 0x481],
	[0x48a, 0x52f],
	[0x531, 0x556],
	[0x559, 0x559],
	[0x560, 0x588],
	[0x5d0, 0x5ea],
	[0x5ef, 0x5f2],
	[0x620, 0x64a],
	[0x66e, 0x66f],
	[0x671, 0x6d3],
	[0x6d5, 0x6d5],
	[0x6e5, 0x6e6],
	[0x6ee, 0x6ef],
	[0x6fa, 0x6fc],
	[0x6ff, 0x6ff],
	[0x710, 0x710],
	[0x712, 0x72f],
	[0x74d, 0x7a5],
	[0x7b1, 0x7b1],
	[0x7ca, 0x7ea],
	[0x7f4, 0x7f5],
	[0x7fa, 0x7fa],
	[0x900, 0x904],
	[0x905, 0x939],
	[0x93d, 0x93d],
	[0x950, 0x950],
	[0x958, 0x961],
	[0x971, 0x980],
	[0x985, 0x98c],
	[0x98f, 0x990],
	[0x993, 0x9a8],
	[0x9aa, 0x9b0],
	[0x9b2, 0x9b2],
	[0x9b6, 0x9b9],
	[0x9bd, 0x9bd],
	[0x9ce, 0x9ce],
	[0x9dc, 0x9dd],
	[0x9df, 0x9e1],
	[0x9f0, 0x9f1],
	[0x9fc, 0x9fc],
	[0xa05, 0xa0a],
	[0xa0f, 0xa10],
	[0xa13, 0xa28],
	[0xa2a, 0xa30],
	[0xa32, 0xa33],
	[0xa35, 0xa36],
	[0xa38, 0xa39],
	[0xa59, 0xa5c],
	[0xa5e, 0xa5e],
	[0xa72, 0xa74],
	[0xa85, 0xa8d],
	[0xa8f, 0xa91],
	[0xa93, 0xaa8],
	[0xaaa, 0xab0],
	[0xab2, 0xab3],
	[0xab5, 0xab9],
	[0xabd, 0xabd],
	[0xad0, 0xad0],
	[0xae0, 0xae1],
	[0xaf9, 0xaf9],
	[0xb05, 0xb0c],
	[0xb0f, 0xb10],
	[0xb13, 0xb28],
	[0xb2a, 0xb30],
	[0xb32, 0xb33],
	[0xb35, 0xb39],
	[0xb3d, 0xb3d],
	[0xb5c, 0xb5d],
	[0xb5f, 0xb61],
	[0xb71, 0xb71],
	[0xb83, 0xb83],
	[0xb85, 0xb8a],
	[0xb8e, 0xb90],
	[0xb92, 0xb95],
	[0xb99, 0xb9a],
	[0xb9c, 0xb9c],
	[0xb9e, 0xb9f],
	[0xba3, 0xba4],
	[0xba8, 0xbaa],
	[0xbae, 0xbb9],
	[0xbd0, 0xbd0],
	[0xc05, 0xc0c],
	[0xc0e, 0xc10],
	[0xc12, 0xc28],
	[0xc2a, 0xc39],
	[0xc3d, 0xc3d],
	[0xc58, 0xc5a],
	[0xc5d, 0xc5d],
	[0xc60, 0xc61],
	[0xc80, 0xc80],
	[0xc85, 0xc8c],
	[0xc8e, 0xc90],
	[0xc92, 0xca8],
	[0xcaa, 0xcb3],
	[0xcb5, 0xcb9],
	[0xcbd, 0xcbd],
	[0xcdd, 0xcde],
	[0xce0, 0xce1],
	[0xcf1, 0xcf2],
	[0xd04, 0xd0c],
	[0xd0e, 0xd10],
	[0xd12, 0xd3a],
	[0xd3d, 0xd3d],
	[0xd4e, 0xd4e],
	[0xd54, 0xd56],
	[0xd5f, 0xd61],
	[0xd7a, 0xd7f],
	[0xd9a, 0xdb1],
	[0xdb3, 0xdbb],
	[0xdbd, 0xdbd],
	[0xdc0, 0xdc6],
	[0xe01, 0xe30],
	[0xe32, 0xe33],
	[0xe40, 0xe46],
	[0xe81, 0xe82],
	[0xe84, 0xe84],
	[0xe86, 0xe8a],
	[0xe8c, 0xea3],
	[0xea5, 0xea5],
	[0xea7, 0xeb0],
	[0xeb2, 0xeb3],
	[0xebd, 0xebd],
	[0xec0, 0xec4],
	[0xec6, 0xec6],
	[0xedc, 0xedf],
	[0xf00, 0xf00],
	[0xf40, 0xf47],
	[0xf49, 0xf6c],
	[0xf88, 0xf8c],
	[0x1000, 0x102a],
	[0x103f, 0x103f],
	[0x1050, 0x1055],
	[0x105a, 0x105d],
	[0x1061, 0x1061],
	[0x1065, 0x1066],
	[0x106e, 0x1070],
	[0x1075, 0x1081],
	[0x108e, 0x108e],
	[0x10a0, 0x10c5],
	[0x10c7, 0x10c7],
	[0x10cd, 0x10cd],
	[0x10d0, 0x10fa],
	[0x10fc, 0x10ff],
	[0x1100, 0x1248],
	[0x124a, 0x124d],
	[0x1250, 0x1256],
	[0x1258, 0x1258],
	[0x125a, 0x125d],
	[0x1260, 0x1288],
	[0x128a, 0x128d],
	[0x1290, 0x12b0],
	[0x12b2, 0x12b5],
	[0x12b8, 0x12be],
	[0x12c0, 0x12c0],
	[0x12c2, 0x12c5],
	[0x12c8, 0x12d6],
	[0x12d8, 0x1310],
	[0x1312, 0x1315],
	[0x1318, 0x135a],
	[0x13a0, 0x13f5],
	[0x13f8, 0x13fd],
	[0x1401, 0x166c],
	[0x166e, 0x167f],
	[0x1681, 0x169a],
	[0x16a0, 0x16ea],
	[0x1700, 0x1711],
	[0x171f, 0x1731],
	[0x1740, 0x1751],
	[0x1760, 0x176c],
	[0x176e, 0x1770],
	[0x1780, 0x17b3],
	[0x17d7, 0x17d7],
	[0x17dc, 0x17dc],
	[0x1820, 0x1878],
	[0x1880, 0x18a8],
	[0x18aa, 0x18aa],
	[0x18b0, 0x18f5],
	[0x1900, 0x191e],
	[0x1950, 0x196d],
	[0x1970, 0x1974],
	[0x1980, 0x19ab],
	[0x19b0, 0x19c9],
	[0x1a00, 0x1a16],
	[0x1a20, 0x1a54],
	[0x1aa7, 0x1aa7],
	[0x1b05, 0x1b33],
	[0x1b45, 0x1b4c],
	[0x1b83, 0x1ba0],
	[0x1bae, 0x1baf],
	[0x1bba, 0x1be5],
	[0x1c00, 0x1c23],
	[0x1c4d, 0x1c4f],
	[0x1c5a, 0x1c7d],
	[0x1ce9, 0x1cec],
	[0x1cee, 0x1cf3],
	[0x1cf5, 0x1cf6],
	[0x1d00, 0x1dbf],
	[0x1e00, 0x1e99],
	[0x1e9e, 0x1e9e],
	[0x1ea0, 0x1ef9],
	[0x1f00, 0x1f15],
	[0x1f18, 0x1f1d],
	[0x1f20, 0x1f45],
	[0x1f48, 0x1f4d],
	[0x1f50, 0x1f57],
	[0x1f59, 0x1f59],
	[0x1f5b, 0x1f5b],
	[0x1f5d, 0x1f5d],
	[0x1f5f, 0x1f7d],
	[0x1f80, 0x1fb4],
	[0x1fb6, 0x1fbc],
	[0x1fbe, 0x1fbe],
	[0x1fc2, 0x1fc4],
	[0x1fc6, 0x1fcc],
	[0x1fd0, 0x1fd3],
	[0x1fd6, 0x1fdb],
	[0x1fe0, 0x1fec],
	[0x1ff2, 0x1ff4],
	[0x1ff6, 0x1ffc],
	[0x2071, 0x2071],
	[0x207f, 0x207f],
	[0x2090, 0x209c],
	[0x2102, 0x2102],
	[0x2107, 0x2107],
	[0x210a, 0x2113],
	[0x2115, 0x2115],
	[0x2118, 0x211d],
	[0x2124, 0x2124],
	[0x2126, 0x2126],
	[0x2128, 0x2128],
	[0x212a, 0x212d],
	[0x212f, 0x2139],
	[0x213c, 0x213f],
	[0x2145, 0x2149],
	[0x214e, 0x214e],
	[0x2160, 0x2188],
	[0x2c00, 0x2ce4],
	[0x2ceb, 0x2cf3],
	[0x2d00, 0x2d25],
	[0x2d27, 0x2d27],
	[0x2d2d, 0x2d2d],
	[0x2d30, 0x2d67],
	[0x2d80, 0x2d96],
	[0x2da0, 0x2da6],
	[0x2da8, 0x2dae],
	[0x2db0, 0x2db6],
	[0x2db8, 0x2dbe],
	[0x2dc0, 0x2dc6],
	[0x2dc8, 0x2dce],
	[0x2dd0, 0x2dd6],
	[0x2dd8, 0x2dde],
	[0x3005, 0x3007],
	[0x3021, 0x3029],
	[0x3031, 0x3035],
	[0x3038, 0x303c],
	[0x3041, 0x3096],
	[0x309d, 0x309f],
	[0x30a1, 0x30fa],
	[0x30fc, 0x30ff],
	[0x3105, 0x3127],
	[0x3131, 0x318e],
	[0x31a0, 0x31bf],
	[0x31f0, 0x31ff],
	[0x3400, 0x4dbf],
	[0x4e00, 0xa48f],
	[0xa4d0, 0xa4fd],
	[0xa500, 0xa60c],
	[0xa610, 0xa61f],
	[0xa62a, 0xa62b],
	[0xa640, 0xa66f],
	[0xa674, 0xa67d],
	[0xa67f, 0xa697],
	[0xa717, 0xa71f],
	[0xa722, 0xa788],
	[0xa78b, 0xa7dc],
	[0xa7f1, 0xa7f1],
	[0xa7f5, 0xa7f6],
	[0xab30, 0xab5a],
	[0xab5c, 0xab69],
	[0xab70, 0xabea],
	[0xabec, 0xabed],
	[0xabf0, 0xabf9],
	[0xac00, 0xd7a3],
	[0xd7b0, 0xd7c6],
	[0xd7cb, 0xd7fb],
	[0xf900, 0xfa6d],
	[0xfa70, 0xfad9],
	[0xfb00, 0xfb06],
	[0xfb13, 0xfb17],
	[0xfb1d, 0xfb1d],
	[0xfb1f, 0xfb28],
	[0xfb2a, 0xfb36],
	[0xfb38, 0xfb3c],
	[0xfb3e, 0xfb3e],
	[0xfb40, 0xfb41],
	[0xfb43, 0xfb44],
	[0xfb46, 0xfbb1],
	[0xfbd3, 0xfc5d],
	[0xfc64, 0xfd49],
	[0xfd50, 0xfd8f],
	[0xfd92, 0xfdc7],
	[0xfdf0, 0xfdf9],
	[0xff21, 0xff3a],
	[0xff41, 0xff5a],
	[0xff66, 0xff9d],
	[0xffa0, 0xffbe],
	[0xffc2, 0xffc7],
	[0xffca, 0xffcf],
	[0xffd2, 0xffd7],
	[0xffda, 0xffdc],
	[0x10000, 0x1000b],
	[0x1000d, 0x10026],
	[0x10028, 0x1003a],
	[0x1003c, 0x1003d],
	[0x1003f, 0x1004d],
	[0x10050, 0x1005d],
	[0x10080, 0x100fa],
	[0x10400, 0x1049d],
	[0x104b0, 0x104fb],
	[0x10500, 0x10527],
	[0x10530, 0x10563],
	[0x10c80, 0x10cb2],
	[0x10cc0, 0x10cf2],
	[0x10d00, 0x10d23],
	[0x11003, 0x11037],
	[0x11083, 0x110af],
	[0x110d0, 0x110e8],
	[0x11103, 0x11126],
	[0x11144, 0x11144],
	[0x11150, 0x11172],
	[0x11176, 0x11176],
	[0x11183, 0x111b2],
	[0x111c1, 0x111c4],
	[0x11200, 0x11211],
	[0x11213, 0x1122b],
	[0x11305, 0x1130c],
	[0x1130f, 0x11310],
	[0x11313, 0x11328],
	[0x1132a, 0x11330],
	[0x11332, 0x11333],
	[0x11335, 0x11339],
	[0x11600, 0x1162f],
	[0x11680, 0x116aa],
	[0x11700, 0x1171a],
	[0x11800, 0x1182b],
	[0x118a0, 0x118df],
	[0x11900, 0x11906],
	[0x11909, 0x11909],
	[0x1190c, 0x11913],
	[0x11915, 0x11916],
	[0x11918, 0x1192f],
	[0x11a00, 0x11a32],
	[0x11a3a, 0x11a3a],
	[0x11a50, 0x11a50],
	[0x11a5c, 0x11a83],
	[0x11a86, 0x11a89],
	[0x11a9a, 0x11aa2],
	[0x11c00, 0x11c08],
	[0x11c0a, 0x11c2e],
	[0x11c72, 0x11c8f],
	[0x11d00, 0x11d06],
	[0x11d08, 0x11d09],
	[0x11d0b, 0x11d30],
	[0x11d60, 0x11d65],
	[0x11d67, 0x11d68],
	[0x11d6a, 0x11d89],
	[0x12000, 0x12399],
	[0x12480, 0x12543],
	[0x12f90, 0x12ff0],
	[0x13000, 0x1342e],
	[0x14400, 0x14646],
	[0x16800, 0x16a38],
	[0x16a40, 0x16a5e],
	[0x16a70, 0x16abe],
	[0x16b00, 0x16b36],
	[0x16f00, 0x16f4a],
	[0x16f50, 0x16f50],
	[0x17000, 0x187ec],
	[0x18800, 0x18af2],
	[0x1b000, 0x1b122],
	[0x1b150, 0x1b152],
	[0x1b164, 0x1b167],
	[0x1b170, 0x1b2fb],
	[0x1d400, 0x1d454],
	[0x1d456, 0x1d49c],
	[0x1d49e, 0x1d49f],
	[0x1d4a2, 0x1d4a2],
	[0x1d4a5, 0x1d4a6],
	[0x1d4a9, 0x1d4ac],
	[0x1d4ae, 0x1d4b9],
	[0x1d4bb, 0x1d4bb],
	[0x1d4bd, 0x1d4c3],
	[0x1d4c5, 0x1d505],
	[0x1d507, 0x1d50a],
	[0x1d50d, 0x1d514],
	[0x1d516, 0x1d51c],
	[0x1d51e, 0x1d539],
	[0x1d53b, 0x1d53e],
	[0x1d540, 0x1d544],
	[0x1d546, 0x1d546],
	[0x1d548, 0x1d550],
	[0x1d552, 0x1d6a5],
	[0x1e900, 0x1e943],
	[0x1e94b, 0x1e94b],
	[0x20000, 0x2a6df],
	[0x2a700, 0x2b81d],
	[0x2b820, 0x2cea1],
	[0x2ceb0, 0x2ebe0],
	[0x2f800, 0x2fa1d],
	[0x30000, 0x3134f],
]

// bpe_number_ranges covers \p{N} (decimal digits across scripts; letter
// numbers like Roman numerals stay punct-adjacent per the pattern).
const bpe_number_ranges = [
	[0x30, 0x39],
	[0x660, 0x669],
	[0x6f0, 0x6f9],
	[0x7c0, 0x7c9],
	[0x966, 0x96f],
	[0x9e6, 0x9ef],
	[0xa66, 0xa6f],
	[0xae6, 0xaef],
	[0xb66, 0xb6f],
	[0xbe6, 0xbef],
	[0xc66, 0xc6f],
	[0xce6, 0xcef],
	[0xd66, 0xd6f],
	[0xde6, 0xdef],
	[0xe50, 0xe59],
	[0xed0, 0xed9],
	[0xf20, 0xf29],
	[0x1040, 0x1049],
	[0x1090, 0x1099],
	[0x17e0, 0x17e9],
	[0x1810, 0x1819],
	[0x1946, 0x194f],
	[0x19d0, 0x19d9],
	[0x1a80, 0x1a89],
	[0x1a90, 0x1a99],
	[0x1b50, 0x1b59],
	[0x1bb0, 0x1bb9],
	[0x1c40, 0x1c49],
	[0x1c50, 0x1c59],
	[0xa620, 0xa629],
	[0xa8d0, 0xa8d9],
	[0xa900, 0xa909],
	[0xa9d0, 0xa9d9],
	[0xaa50, 0xaa59],
	[0xabf0, 0xabf9],
	[0xff10, 0xff19],
	[0x104a0, 0x104a9],
	[0x11066, 0x1106f],
	[0x110f0, 0x110f9],
	[0x11136, 0x1113f],
	[0x111d0, 0x111d9],
	[0x112f0, 0x112f9],
	[0x11450, 0x11459],
	[0x114d0, 0x114d9],
	[0x11650, 0x11659],
	[0x116c0, 0x116c9],
	[0x11730, 0x11739],
	[0x118e0, 0x118e9],
	[0x11c50, 0x11c59],
	[0x11d50, 0x11d59],
	[0x11da0, 0x11da9],
	[0x16a60, 0x16a69],
	[0x16b50, 0x16b59],
	[0x1d7ce, 0x1d7ff],
	[0x1e140, 0x1e149],
	[0x1e950, 0x1e959],
]

// bpe_space_ranges covers \s beyond ASCII (NBSP, Ogham space, Zs blocks,
// line/paragraph separators, narrow NBSP, ideographic space).
const bpe_space_ranges = [
	[0x09, 0x0d],
	[0x20, 0x20],
	[0x85, 0x85],
	[0xa0, 0xa0],
	[0x1680, 0x1680],
	[0x2000, 0x200a],
	[0x2028, 0x2029],
	[0x202f, 0x202f],
	[0x205f, 0x205f],
	[0x3000, 0x3000],
]

fn bpe_in_ranges(r u32, ranges [][]int) bool {
	mut lo := 0
	mut hi := ranges.len
	ri := int(r)
	for lo < hi {
		mid := (lo + hi) / 2
		if ri < ranges[mid][0] {
			hi = mid
		} else if ri > ranges[mid][1] {
			lo = mid + 1
		} else {
			return true
		}
	}
	return false
}

fn bpe_is_letter(r u32) bool {
	return bpe_in_ranges(r, bpe_letter_ranges)
}

fn bpe_is_number(r u32) bool {
	return bpe_in_ranges(r, bpe_number_ranges)
}

fn bpe_is_space(r u32) bool {
	return bpe_in_ranges(r, bpe_space_ranges)
}

// bpe_pretokenize splits text per the Qwen2 Split pattern (isolated pieces:
// merges never cross piece boundaries). Hand-rolled: V's regex lacks the
// \p classes and lookahead this pattern needs, and the kernel port wants
// zero-dependency code anyway.
fn bpe_pretokenize(text string) []string {
	mut out := []string{}
	mut i := 0
	n := text.len
	for i < n {
		r, w := bpe_rune_at(text, i)
		// 1. contractions: 's 't 're 've 'm 'll 'd (case-insensitive)
		if r == u32(`'`) {
			rest := bpe_match_contraction(text, i)
			if rest > i {
				out << text[i..rest]
				i = rest
				continue
			}
		}
		// 2. [^\r\n\p{L}\p{N}]?\p{L}+ : optional stray char + letter run
		if bpe_is_letter(r) {
			mut j := i + w
			for j < n {
				r2, w2 := bpe_rune_at(text, j)
				if !bpe_is_letter(r2) {
					break
				}
				j += w2
			}
			out << text[i..j]
			i = j
			continue
		}
		if r != u32(`\r`) && r != u32(`\n`) && !bpe_is_letter(r) && !bpe_is_number(r) {
			if i + w < n {
				r2, w2 := bpe_rune_at(text, i + w)
				if bpe_is_letter(r2) {
					mut j := i + w + w2
					for j < n {
						r3, w3 := bpe_rune_at(text, j)
						if !bpe_is_letter(r3) {
							break
						}
						j += w3
					}
					out << text[i..j]
					i = j
					continue
				}
			}
		}
		// 3. \p{N} : single number rune (Qwen splits digits one by one)
		if bpe_is_number(r) {
			out << text[i..i + w]
			i += w
			continue
		}
		// 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*` : optional space + punct run
		if r == u32(` `) || (!bpe_is_space(r) && !bpe_is_letter(r) && !bpe_is_number(r)
			&& r != u32(`\r`) && r != u32(`\n`)) {
			mut j := i
			if text[j] == ` ` {
				j++
			}
			for j < n {
				r2, w2 := bpe_rune_at(text, j)
				if bpe_is_space(r2) || bpe_is_letter(r2) || bpe_is_number(r2) || r2 == u32(`\r`)
					|| r2 == u32(`\n`) {
					break
				}
				j += w2
			}
			// trailing [\r\n]*
			for j < n && (text[j] == `\r` || text[j] == `\n`) {
				j++
			}
			// a lone leading space with no punct after it falls through below
			if j > i + (if text[i] == ` ` {
				1
			} else {
				0
			}) {
				out << text[i..j]
				i = j
				continue
			}
		}
		// 5/6/7. whitespace runs. \s*[\r\n]+ takes spaces + newlines (stops
		// after the last newline); \s+(?!\S) backtracks to the longest space
		// prefix followed by space-or-end (same as llama.cpp's engine);
		// a lone space before other text falls through to \s+.
		if bpe_is_space(r) || r == u32(`\r`) || r == u32(`\n`) {
			mut j := i
			mut last_nl := -1
			for j < n {
				r2, w2 := bpe_rune_at(text, j)
				if r2 == u32(`\r`) || r2 == u32(`\n`) {
					last_nl = j + w2
					j += w2
				} else if bpe_is_space(r2) {
					j += w2
				} else {
					break
				}
			}
			if last_nl >= 0 {
				out << text[i..last_nl]
				i = last_nl
				continue
			}
			k := j - i
			if j >= n {
				// trailing spaces at end: \s+(?!\S) takes all.
				out << text[i..j]
				i = j
				continue
			}
			// spaces followed by non-space: backtrack (k-1 spaces when
			// k > 1; a lone space goes whole to \s+).
			if k > 1 {
				out << text[i..j - 1]
				i = j - 1
				continue
			}
			out << text[i..j]
			i = j
			continue
		}
		// fallback: single rune (should be unreachable)
		out << text[i..i + w]
		i += w
	}
	return out
}

// bpe_match_contraction matches 's 't 're 've 'm 'll 'd case-insensitively,
// returning the end offset or -1. Mirrors (?i:'s|'t|'re|'ve|'m|'ll|'d).
fn bpe_match_contraction(text string, off int) int {
	// text[off] == '\'' guaranteed by caller
	if off + 1 >= text.len {
		return -1
	}
	mut c1 := text[off + 1]
	if c1 >= `A` && c1 <= `Z` {
		c1 += 32
	}
	if c1 == `s` || c1 == `t` || c1 == `m` || c1 == `d` {
		return off + 2
	}
	if off + 2 >= text.len {
		return -1
	}
	mut c2 := text[off + 2]
	if c2 >= `A` && c2 <= `Z` {
		c2 += 32
	}
	if (c1 == `r` && c2 == `e`) || (c1 == `v` && c2 == `e`) || (c1 == `l` && c2 == `l`) {
		return off + 3
	}
	return -1
}
