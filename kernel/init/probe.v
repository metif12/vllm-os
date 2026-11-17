// kernel/init/probe.v — Phase 4 step-1 smoke test: proves a normal V
// userland binary (GC, strings, println) runs on the vinix-mini fork.
// Built in WSL: v -cc gcc -cflags '-static' -o sbin/vllm probe.v
// (placed at /sbin/vllm in the initramfs; the C init execve's it after
// printing boot ok).
// Serial must show: `[vllm-os] userland V probe ok` (stdout is the
// framebuffer console, invisible on -display none, so also mirror to
// /dev/com1 explicitly).
import os

fn main() {
	msg := '[vllm-os] userland V probe ok\n'
	print(msg)
	os.write_file('/dev/com1', msg) or { eprintln('probe: no com1: ${err}') }
}
