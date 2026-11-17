// kernel/init/init.c — vinix-mini first userland (/sbin/init).
//
// Raw Linux syscalls only, no libc: `gcc -static -nostdlib -O2 -o init init.c`.
// Prints `[vllm-os] boot ok`, listens on :8000, answers every connection
// with a stub JSON body. This closes the Phase-3 boot gate (serial shows
// boot ok, TCP SYN on :8000 answered); the inference server replaces the
// reply path in Phase 4.
//
// Syscalls used — write socket setsockopt bind listen accept read close
// exit_group — all present in kernel/mini's Linux-compat table
// (syscall/table/linux_table_amd64.v: 1/41/54/49/50/43/231).
typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

static long sc1(long n, long a) {
	long r;
	__asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a) : "rcx", "r11", "memory");
	return r;
}

static long sc2(long n, long a, long b) {
	long r;
	__asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b) : "rcx", "r11", "memory");
	return r;
}

static long sc3(long n, long a, long b, long c) {
	long r;
	__asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c) : "rcx", "r11", "memory");
	return r;
}

static long sc5(long n, long a, long b, long c, long d, long e) {
	register long r10 __asm__("r10") = d;
	register long r8 __asm__("r8") = e;
	long r;
	__asm__ volatile("syscall"
					 : "=a"(r)
					 : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8)
					 : "rcx", "r11", "memory");
	return r;
}

static u64 slen(const char *s) {
	u64 n = 0;
	while (s[n] != '\0') {
		n++;
	}
	return n;
}

static void puts1(const char *s) {
	sc3(1 /* write */, 1, (long)s, (long)slen(s));
}

// u16 -> network byte order without libc htons.
static u16 be16(u16 v) {
	return (u16)(((v & 0xFFu) << 8) | ((v >> 8) & 0xFFu));
}

struct sockaddr_in {
	u16 family;
	u16 port_be;
	u32 addr_be;
	u8 zero[8];
};

static void fail(const char *msg) {
	puts1(msg);
	sc1(231 /* exit_group */, 1);
}

void _start(void) {
	puts1("[vllm-os] boot ok\n");

	long fd = sc3(41 /* socket */, 2 /* AF_INET */, 1 /* SOCK_STREAM */, 0);
	if (fd < 0) {
		fail("init: socket failed\n");
	}
	u32 one = 1;
	sc5(54 /* setsockopt */, fd, 1 /* SOL_SOCKET */, 2 /* SO_REUSEADDR */, (long)&one, 4);

	struct sockaddr_in sa;
	sa.family = 2; // AF_INET
	sa.port_be = be16(8000);
	sa.addr_be = 0; // INADDR_ANY
	for (u64 i = 0; i < sizeof(sa.zero); i++) {
		sa.zero[i] = 0;
	}
	if (sc3(49 /* bind */, fd, (long)&sa, (long)sizeof(sa)) < 0) {
		fail("init: bind :8000 failed\n");
	}
	if (sc2(50 /* listen */, fd, 16) < 0) {
		fail("init: listen failed\n");
	}
	puts1("[vllm-os] listening on :8000\n");

	static const char body[] = "{\"model\":\"mini-stub\",\"boot\":\"ok\"}";
	static const char hdr[] = "HTTP/1.1 200 OK\r\n"
							  "content-type: application/json\r\n"
							  "content-length: 33\r\n"
							  "connection: close\r\n"
							  "\r\n";
	// hdr claims 33: assert at runtime, refuse to serve a lying header.
	if (slen(body) != 33) {
		fail("init: stub body length drifted from header\n");
	}

	char req[1024];
	for (;;) {
		long c = sc3(43 /* accept */, fd, 0, 0);
		if (c < 0) {
			continue;
		}
		sc3(0 /* read */, c, (long)req, (long)sizeof(req));
		sc3(1 /* write */, c, (long)hdr, (long)slen(hdr));
		sc3(1 /* write */, c, (long)body, 33);
		sc1(3 /* close */, c);
	}
}
