module infer

// Phase-1 whitespace tokenizer stub with the Qwen chat-template helper.
// Real BPE (151k vocab) is a pure-V port in phase 2; the API shape stays.

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

// Qwen3 chat template (ChatML-style). Applied server-side.
pub fn apply_chat_template(messages []ChatMsg) string {
	mut s := ''
	for m in messages {
		s += '<|im_start|>${m.role}\n${m.content}<|im_end|>\n'
	}
	s += '<|im_start|>assistant\n'
	return s
}

pub struct ChatMsg {
pub:
	role    string
	content string
}
