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

**Stage 0: scaffolding.** The ELF64 header and program-header tables parse, with tests. Nothing executes yet.

**Tests:** 8 There is no
translator, no WASM emitter and no syscall layer in this repository today.

```bash
npm ci && npm run verify
```

## Roadmap

1. **ELF loading** (done): ELF64 headers, `PT_LOAD` segments, explicit refusals for 32-bit, big-endian, relocatable.
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
