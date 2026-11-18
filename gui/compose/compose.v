module compose

// Pure ISO-builder logic: validation + command generation. No ui2
// dependency, so `v test gui/compose` runs on any V (including the
// pinned release); the ui2 shell (gui/main.v) needs a json2-capable V.
pub struct IsoForm {
pub mut:
	arch      string = 'x86_64'
	model     string
	tokenizer string
	stub      bool
	port      string = '8000'
	ctx       string = '4096'
	backend   string = 'cpu'
	preset    string = 'balanced'
	temp      string
	mem       string = '4'
	auth      string
}

pub const backends = ['cpu', 'cpu-blocked', 'cpu-simd', 'cpu-parallel', 'auto']
pub const presets = ['deterministic', 'precise', 'balanced', 'creative']
const overhead_bytes = u64(1) << 30

fn is_decimal(s string) bool {
	mut digits := 0
	for ch in s {
		if ch >= `0` && ch <= `9` {
			digits++
		} else if ch != `.` && ch != `+` && ch != `-` && ch != `e` && ch != `E` {
			return false
		}
	}
	return digits > 0
}

// validate_iso_form returns '' when the form is buildable, else the
// reason. Mirrors scripts/build_iso.py refusals one for one.
pub fn validate_iso_form(f IsoForm, model_bytes u64) string {
	if f.arch != 'x86_64' {
		return 'refused: only x86_64 boots (arm64 drivers not ported)'
	}
	if f.model == '' {
		return 'model path required'
	}
	port := f.port.int()
	if port < 1 || port > 65535 {
		return 'port out of range 1..65535'
	}
	ctx := f.ctx.int()
	if ctx < 128 || ctx > 131072 {
		return 'ctx out of range 128..131072'
	}
	if f.backend !in backends {
		return 'unknown backend (one of cpu, cpu-blocked, cpu-simd, cpu-parallel, auto)'
	}
	if f.preset !in presets {
		return 'unknown sampler preset (one of deterministic, precise, balanced, creative)'
	}
	if f.temp != '' {
		if !is_decimal(f.temp) {
			return 'temp is not a number'
		}
		t := f.temp.f32()
		if t < 0.0 || t > 2.0 {
			return 'temp out of range 0..2'
		}
	}
	mem := f.mem.int()
	if mem < 1 || mem > 64 {
		return 'guest RAM must be 1..64 GiB'
	}
	if model_bytes > 0 && model_bytes + overhead_bytes > u64(mem) * overhead_bytes {
		return 'weights + 1 GiB runtime exceed the guest budget'
	}
	return ''
}

fn quote_sh(s string) string {
	return "'" + s.replace("'", "'\\''") + "'"
}

// build_iso_command renders the exact scripts/build_iso.py invocation.
// Call only after validate_iso_form returns ''.
pub fn build_iso_command(f IsoForm) string {
	mut cmd := 'python3 scripts/build_iso.py --model ' + f.model
	if f.tokenizer != '' {
		cmd += ' --tokenizer ' + f.tokenizer
	}
	if f.stub {
		cmd += ' --allow-stub-tokenizer'
	}
	cmd += ' --port ' + f.port.int().str()
	cmd += ' --ctx ' + f.ctx.int().str()
	cmd += ' --backend ' + f.backend
	cmd += ' --preset ' + f.preset
	if f.temp != '' {
		cmd += ' --temp ' + f.temp.f32().str()
	}
	cmd += ' --mem ' + f.mem.int().str()
	if f.auth != '' {
		cmd += ' --auth-token ' + quote_sh(f.auth)
	}
	return cmd
}

// boot_command renders the QEMU line for the built image.
pub fn boot_command(f IsoForm) string {
	return 'qemu-system-x86_64 -cdrom vllm-os.iso -m ' + f.mem.int().str() + 'G -display none -serial stdio ' + '-netdev user,id=n0,hostfwd=tcp::18000-:' + f.port.int().str() + ' -device e1000,netdev=n0'
}
