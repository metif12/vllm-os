module formats

// ContainerHandler is the extension point for weight containers. A new
// format = one struct implementing this interface + one line appending it
// to default_handlers(). The generic loader path in infer (load_model)
// dispatches through handler_for, so format #4 touches no existing code.
//
// GGUF is intentionally NOT here: it needs its rich index (GgmlType,
// block offsets in infer.GgufModel), which ModelFile cannot express.
// GGUF keeps its native path in infer; everything else goes generic.
pub interface ContainerHandler {
	// format reports the WeightFormat this handler loads.
	format() WeightFormat
	// sniff_prefix reports whether a byte prefix identifies this container.
	// Pure function of the bytes — used by detect_with for magic sniffing.
	sniff_prefix(prefix []u8) bool
	// load_index parses the header-only index (no blob reads).
	load_index(path string) !ModelFile
}

// SafetensorsHandler loads SafeTensors header-only indexes.
pub struct SafetensorsHandler {}

// format reports the SafeTensors container kind.
pub fn (h SafetensorsHandler) format() WeightFormat {
	return .safetensors
}

// sniff_prefix matches the u64 header length + '{"' JSON opening.
pub fn (h SafetensorsHandler) sniff_prefix(prefix []u8) bool {
	if prefix.len < 10 {
		return false
	}
	mut n := u64(0)
	for i in 0 .. 8 {
		n |= u64(prefix[i]) << u64(i * 8)
	}
	return n > 2 && n < 256 * 1024 * 1024 && prefix[8] == `{` && prefix[9] == `"`
}

// load_index parses the SafeTensors header index (no blob reads).
pub fn (h SafetensorsHandler) load_index(path string) !ModelFile {
	return load_safetensors(path)
}

// PytorchHintHandler detects legacy PyTorch checkpoints and fails with a
// convert-first error. A real .bin loader can replace this struct without
// touching dispatch: same interface, same registry slot.
pub struct PytorchHintHandler {}

// format reports the PyTorch-hint container kind.
pub fn (h PytorchHintHandler) format() WeightFormat {
	return .pytorch_bin
}

// sniff_prefix never matches: PyTorch is detected by extension only.
pub fn (h PytorchHintHandler) sniff_prefix(prefix []u8) bool {
	// No reliable magic: matched by extension in detect_with.
	return false
}

// load_index always fails: PyTorch checkpoints must be converted first.
pub fn (h PytorchHintHandler) load_index(path string) !ModelFile {
	return error('pytorch .bin is detected but not loadable: convert to safetensors or GGUF first (scripts/fetch_model.sh)')
}

// default_handlers lists every generic container handler. Append new
// formats here — dispatch (handler_for, detect_with) needs no edits.
pub fn default_handlers() []ContainerHandler {
	return [ContainerHandler(SafetensorsHandler{}), ContainerHandler(PytorchHintHandler{})]
}

// handler_for returns the registered handler for a format, or none.
pub fn handler_for(fmt WeightFormat) ?ContainerHandler {
	for h in default_handlers() {
		if h.format() == fmt {
			return h
		}
	}
	return none
}

// detect_with sniffs the container using handler magic first, then the
// PyTorch extension fallback. GGUF magic is matched inline (its native
// loader lives in infer, not behind this interface — see above).
pub fn detect_with(path string, handlers []ContainerHandler) !WeightFormat {
	data := read_prefix(path, 16)!
	if data.len >= 4 && data[0] == `G` && data[1] == `G` && data[2] == `U` && data[3] == `F` {
		return .gguf
	}
	for h in handlers {
		if h.sniff_prefix(data) {
			return h.format()
		}
	}
	lower := path.to_lower()
	if lower.ends_with('.bin') || lower.ends_with('.pt') || lower.ends_with('.pth') {
		return .pytorch_bin
	}
	return .unknown
}
