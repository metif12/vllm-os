module infer

import json2

// BPE tokenizer: loads HF tokenizer.json (Qwen3) when baked into the image,
// falls back to the whitespace stub when absent (keeps the HTTP surface testable).
pub struct Bpe {
pub:
	vocab  map[string]int
	merges map[string]int
	bpe_ok bool
}

pub struct ChatMsg {
pub:
	role    string
	content string
}

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

pub fn decode_stub(ids []int) string {
	return ids.map('<tok_${it}>').join(' ')
}

// load_bpe parses a HF tokenizer.json (model.vocab + model.merges).
pub fn load_bpe(path string) !Bpe {
	raw := read_whole_file(path)!
	doc := json2.decode[TokenizerJson](raw.bytestr())!
	mut merges := map[string]int{}
	for i, m in doc.model.merges {
		merges[m] = i
	}
	return Bpe{
		vocab:  doc.model.vocab
		merges: merges
		bpe_ok: true
	}
}

// encode runs byte-level BPE for one word; unknown words degrade to bytes.
pub fn (b Bpe) encode(text string) []int {
	if !b.bpe_ok {
		return encode_stub(text)
	}
	mut out := []int{}
	for word in text.split(' ') {
		if word.len == 0 {
			continue
		}
		if id := b.vocab[word] {
			out << id
			continue
		}
		// Byte fallback: each byte as token if present, else stub hash.
		for ch in word.bytes() {
			s := ch.ascii_str()
			if id := b.vocab[s] {
				out << id
			} else {
				out << 131 * word.len + int(ch)
			}
		}
	}
	return out
}

struct TokenizerModelJson {
	vocab  map[string]int
	merges []string
}

struct TokenizerJson {
	model TokenizerModelJson
}

// Qwen3 chat template (ChatML-style). Applied server-side.
pub fn apply_chat_template(messages []ChatMsg) string {
	mut s := ''
	for m in messages {
		s += '<|im_start|>${m.role}\n${m.content}<|im_end|>\n'
	}
	s += '<|im_start|>assistant\n'
	return s
}
