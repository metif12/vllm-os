/* vinix-mini link script: higher-half kernel, 2 MB page alignment so the
 * weights window (1.1 GB mmap) costs the minimum TLB footprint.
 * Used by: v -gc none -freestanding (see scripts/build.sh bare-metal).
 * Sections ordered for boot speed: text first (faulted in linearly),
 * rodata next, then writable data/bss. No debug info in the image. */
OUTPUT_FORMAT(elf64-x86-64)
OUTPUT_ARCH(i386:x86-64)
ENTRY(_start)

BASE = 0xffffffff80000000;
ALIGN_PAGES = 0x200000;

SECTIONS {
    . = BASE + SIZEOF_HEADERS;

    .text ALIGN(ALIGN_PAGES) : AT(ADDR(.text) - BASE) {
        *(.text.boot)   /* _start + early paging, must stay first */
        *(.text .text.*)
    }

    .rodata ALIGN(0x1000) : AT(ADDR(.rodata) - BASE) {
        *(.rodata .rodata.*)
        /* baked model index (tensor table snapshot) goes here, NOT weights */
    }

    .data ALIGN(0x1000) : AT(ADDR(.data) - BASE) {
        *(.data .data.*)
    }

    .bss ALIGN(0x1000) : AT(ADDR(.bss) - BASE) {
        *(COMMON)
        *(.bss .bss.*)
        /* KV-cache arena + net rings live here, zeroed once at boot */
    }

    /DISCARD/ : {
        *(.comment .note* .debug* .eh_frame)
    }
}
