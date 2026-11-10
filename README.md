# elfwasm

**User-mode ELF to WASM translation, with Linux syscalls answered by JavaScript shims.**

Run a normal static Linux binary in a browser tab without emulating a whole PC. This is QEMU-user's idea, retargeted at
WASM: translate the instructions ahead of time, and satisfy `read`/`write`/`mmap`/`exit_group` in JS instead of booting a
kernel.

## Why this shape

[v86](https://github.com/copy/v86) proves x86-to-WASM works, but it emulates a machine: firmware, devices, a boot
sequence. That is why it takes minutes to reach a shell. A static musl binary needs none of that. It needs its segments
mapped, its instructions translated, and about twenty syscalls.

The narrow target is the one that matters here:

| dimension | target |
| --- | --- |
| binary | statically linked musl, ELF64 |
| arch | x86-64 first, aarch64 second |
| threads | single-threaded |
| process | one, no `fork`, no `execve` |
| syscalls | a named allowlist; everything else refuses with `ENOSYS` |

## What it unlocks

A JS-shimmed Node runtime cannot execute native addons: there is nothing to run a `.node` binary on. This does. That
makes it the missing piece between a shimmed runtime and running real `esbuild`, real `sharp`, real `node` itself.

## Status

**Stage 1: loading, not executing.** The ELF64 header, program-header tables, section headers, symbol tables and
`RELA` relocations parse, and the `PT_LOAD` segments are now placed into a WASM linear memory. Nothing executes yet:
there is no decoder, no translator, no WASM emitter and no syscall layer in this repository today.

Refusals are named, not generic, and that is the point: a 32-bit, big-endian, stripped, or 16-byte-symbol-entry object is
rejected with a reason, because a parser that guesses an offset produces confident nonsense downstream.

Relocations (`RELA`) parse too, with their symbol resolved by name and the addend kept signed. `REL`, whose addend lives
in memory rather than in the file, is refused by name, because reading it as `RELA` would relocate to the wrong address
instead of failing.

Segment mapping produces a `MemoryImage`: the base is the lowest segment address rounded down to a WASM page, each
segment records both its virtual address and its byte offset in the buffer, and `loadImage` copies file bytes then
zero-fills every `p_memsz` tail. That zero-fill is explicit rather than assumed, because a fresh `WebAssembly.Memory`
starts zeroed and hides the bug on the first run. Mappings are checked first: overlapping segments, `p_filesz` greater
than `p_memsz`, a segment past EOF, an address not congruent to its file offset modulo `p_align`, and an entry point
outside every executable segment are all refused by name. `toWasmAddress` refuses an unmapped virtual address rather
than returning whatever byte happens to live at that offset.

Verified against `/bin/ls` on the host: 4 segments, 3 WASM pages (192 KiB), 143,353 bytes copied, 4,840 bytes of BSS
zero-filled, entry `0x61d0` translated to WASM offset `0x61d0` holding `0x31`, the first byte of real x86-64 code.

What does **not** work yet: `REL` relocations, extended section numbering (`e_shnum == 0` with the real count in the first
section's `sh_size`), instruction decoding, translation to WASM, and any syscall. Two honest limits of the mapping:
WASM linear memory has no per-page protection, so the `read`/`write`/`execute` flags are carried for the translator to
honour rather than enforced by hardware; and the 32-bit address space means an image larger than 4 GiB is refused.

**Tests:** 36

```bash
npm ci && npm run verify
```

## Roadmap

1. **ELF loading** (done): ELF64 headers, `PT_LOAD` segments mapped into a WASM `MemoryImage`, `RELA` relocations, and
   explicit refusals for 32-bit, big-endian and relocatable objects.
2. **Decoder**: x86-64 instruction decoding to an IR, with the subset a musl `hello world` needs.
3. **Translator**: IR to WASM, one function per basic block, registers in locals.
4. **Syscall layer**: `write`, `exit_group`, `brk`, `mmap`, then the file syscalls, each behind a capability check.
5. **Evidence**: the same binary run under real Linux and under the translator, stdout and exit code compared.

## Non-goals

- Not a full-PC emulator. No BIOS, no devices, no boot.
- Not a JIT for arbitrary self-modifying code initially; self-modifying code lands with explicit invalidation or not at
  all.
- No `fork`/`execve`, so no shells and no `npm install` *by this route*.

## Licence

MIT.
