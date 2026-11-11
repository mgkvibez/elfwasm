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

**Stage 1 done, Stage 2 done for the integer subset, Stage 3 started.** The ELF64 header, program-header tables, section headers, symbol tables and `RELA`
relocations parse, the `PT_LOAD` segments are placed into a WASM linear memory, and the x86-64 decoder covers the subset a
statically linked musl `hello world` uses. Code with loops executes: x86-64 bytes are decoded and translated into a structured WASM module that Node runs, and the
number it returns is compared against the number the same bytes compute. Where binutils is installed that comparison is
against the processor itself, not against a hand-written expectation. A real `hello world`, assembled and linked by binutils, now runs
translated: its `write` reaches a shim, its `exit` stops the module, and its stdout and exit status are compared with what
Linux did.

Refusals are named, not generic, and that is the point: a 32-bit, big-endian, stripped, or 16-byte-symbol-entry object is
rejected with a reason, because a parser that guesses an offset produces confident nonsense downstream.

Relocations (`RELA`) parse too, with their symbol resolved by name and the addend kept signed. `REL`, whose addend lives
in memory rather than in the file, is refused by name, because reading it as `RELA` would relocate to the wrong address
instead of failing.

Segment mapping produces a `MemoryImage`: the base is the lowest segment address rounded down to a WASM page unless the caller
chooses one (`base`, which is how a position-independent file gets moved off address zero), each
segment records both its virtual address and its byte offset in the buffer, and `loadImage` copies file bytes then
zero-fills every `p_memsz` tail. That zero-fill is explicit rather than assumed, because a fresh `WebAssembly.Memory`
starts zeroed and hides the bug on the first run. Mappings are checked first: overlapping segments, `p_filesz` greater
than `p_memsz`, a segment past EOF, an address not congruent to its file offset modulo `p_align`, and an entry point
outside every executable segment are all refused by name. `toWasmAddress` refuses an unmapped virtual address rather
than returning whatever byte happens to live at that offset.

Verified against `/bin/ls` on the host: 4 segments, 3 WASM pages (192 KiB), 143,353 bytes copied, 4,840 bytes of BSS
zero-filled, entry `0x61d0` translated to WASM offset `0x61d0` holding `0x31`, the first byte of real x86-64 code. That
binary is position-independent and its 212 relative relocations are the ones a loader applies; its other 117 name symbols a
dynamic linker would resolve, so it is refused rather than half-loaded. See [The loader](#the-loader).

What does **not** work yet: `REL` relocations, extended section numbering (`e_shnum == 0` with the real count in the first
section's `sh_size`), SSE and `x87`, a `call` into libc (its code needs a dynamic linker's symbol resolution, not just
relocations), `%gs` and every segment base but `%fs`, and every syscall but `write`, `exit`, `exit_group` and `arch_prctl`. Two honest limits of the mapping:
WASM linear memory has no per-page protection, so the `read`/`write`/`execute` flags are carried for the translator to
honour rather than enforced by hardware; and the 32-bit address space means an image larger than 4 GiB is refused.

**Tests:** 184

```bash
npm ci && npm run verify
```


## The decoder

`decode` turns code bytes into a flat instruction list: a mnemonic, its operands, and where each instruction starts. The
covered set is what a statically linked musl `hello world` needs in its own code and in the libc it calls:

- moves: `mov` in every register, memory, immediate and RIP-relative form, byte moves included, plus `lea`, `movzx`,
  `movsx` and `movsxd`;
- arithmetic that sets flags: `add`, `sub`, `and`, `or`, `xor`, `cmp`, `test`, `imul`, `neg`, `not`, the arithmetic
  shifts, and the accumulator-immediate forms;
- control: `call`, `ret` with and without its stack-cleanup immediate, the whole conditional-jump family in both the short and near
  encodings, `jmp`, `setcc`, `syscall`, `leave`, `endbr64`, `int3`, `ud2`, `nop`, and `push`/`pop` (`push` with an
  immediate too);
- the sign-extension pair by width: `cwde`/`cdqe` and `cdq`/`cqo`.

ModRM and SIB decode in full, so `[rsp+4*rax+8]` is one operand rather than a convention, and a byte-sized immediate
comes back sign-extended (`sub rcx, -3`, not `253`).

Three details are read from the encoding because guessing them is silent rather than loud. A byte register's name depends
on whether REX is present: code 4 is `ah` without it and `spl` with it. A `0x66` before `0x0f` is the mandatory prefix of
an SSE encoding, not a 16-bit operand size, so it is refused as SSE. And a segment prefix is kept on the memory operand,
because dropping `fs` from `mov %fs:0x28, %rax` would turn a thread-local read into a read of address `0x28`, which is a
plausible looking load of the wrong memory. The linker's alignment padding (`66 2e 0f 1f 84 00`) counts as one ten-byte
`nop`; it was a `0x66` refusal before, which stopped a walk of a real binary at the first function boundary instead of at
real code.

It is tested against binutils rather than against itself. `tests/fixtures/hello.s` is assembled by `as`, its `.text` is
extracted, and `objdump`'s listing is committed beside it by `tools/build-decode-fixture.sh`. The test then requires the
decoder to place every one of the fixture's 68 instructions at the offset objdump placed it, with the same length, so a
wrong instruction length cannot pass: a decoder that is one byte out still returns plausible looking instructions, just
not the ones the compiler emitted, and everything after that point is out of phase.

Everything outside the subset is refused by offset, never skipped: an SSE opcode, a 16-bit operand size, a lock or repeat
prefix, `mul`/`div` (which read and write implicit registers), the rotates, `movzx` from a 16-bit register, a truncated
instruction, and the `lea` form that cannot exist are each named.

Measured on `/bin/ls`, the decoder accounts for the first 57 instructions of `.text` and then refuses at offset `0x10a` on
`66 0f 6f`, which is `movdqa`: the first SSE instruction in glibc's startup path. That refusal is the claim. Every byte
before it is decoded, and the instruction it stops on is named as SSE rather than as a 16-bit operand size, because those
send a reader to different places in the architecture.

## The translator

`translate` takes the decoder's instructions and emits one WASM function per x86-64 function: no parameters, one `i64`
result, sixteen `i64` globals for the sixteen registers in encoding order. The subset is `mov`, `lea`, `add`, `sub`, `xor`,
`and`, `or`, `imul` at 32 and 64 bits, `cmp` and `test`, the ten conditional branches that need only zf, sf, cf and of,
`jmp` in both directions, loops, `call`, `syscall`, and `ret`, which returns `rax`.

Flags are never materialised as variables. `cmp` and `test` leave their operands in two extra locals and a `jcc` becomes
the comparison itself, at the width the operands were: that is why `cmp $-1, %eax ; jl` and `cmp $-1, %rax ; jl` behave
differently without either being special-cased, and why `jl` after `test %eax,%eax` is the same instruction as `js` here,
which is what the architecture defines it to be. `jo`, `jno`, `jp` and `jnp` are refused, because overflow and parity are
not computed and a branch on an uncomputed flag is a wrong answer.

Branches become structure. WASM has no jump, so a forward conditional branch is an `if` around the bytes it skips with the
condition negated, a branch to the end of a region it sits in is `br_if` at the depth that reaches it, and a backward
branch is a `loop` with `br_if 0` re-entering it. A loop that contains a branch to the instruction just after it is
wrapped in a `block`, and that branch becomes `br_if 1`: out of the loop, out of the block, and on.

The compiler's if/else is recognised as one shape rather than translated branch by branch, and the reason is what a branch
can reach: a conditional jump is an `if` around the code it skips, but a `jmp` out of that code has to leave the whole
construct, which plain nesting cannot express. Translating the pattern as a single `if`/`else` whose end is the join fixed
the case the naive version got wrong, where the else arm ran after a then arm that had already jumped past it.

The cost is generality, and it is refused rather than fudged: a jump into the middle of a branch, or a branch leaving a
loop for somewhere that is not just past it, is irreducible control flow and cannot be expressed by nesting.

Two decisions are load-bearing rather than incidental. A 32-bit operand names the 64-bit local and carries width 32, and
a 32-bit write narrows and widens through `i32` so the upper half is zeroed: that is why `mov $-1, %eax` leaves `rax` at
`0xffffffff` and `add $2, %eax` then yields `1` rather than `0x100000001`. And the module's section lengths are computed
from the bytes they contain, because a stale length is the one mistake in this format that produces a module which loads
and misbehaves instead of one that fails to load.

### Memory

A memory operand names a virtual address, so translating one is a matter of resolving that address against the mapped
image before any WASM is emitted. `translate` takes `addresses` (from `image.addressSpaceOf`) and, for RIP-relative
operands, the virtual address of the first byte it was given: a `[rip + disp32]` operand is measured from that
instruction's own end, exactly as the processor measures it. Both are resolved up front, in one pass, so a branch's
recursion never re-derives an address.

Loads and stores become real `i32.load`/`i64.load`/`i32.store`/`i64.store` with the resolved offset baked in, plus
`i32.store8`/`i32.store16` for the narrower stores. A 32-bit load zero-extends and a 64-bit one does not, which is why
`movl $-1, 0x400100 ; mov 0x400100, %rax` leaves `0xffffffff` and the same pair with `movq` leaves `-1`. The store width
comes from the source operand's register, or, for an immediate, from the opcode the decoder recorded on the destination:
guessing four bytes where one was meant would corrupt three neighbours and validate perfectly.

The module *imports* its memory rather than declaring one. A module that owned its memory would have a buffer the loader
never filled, so every address would read zero while looking legitimate; importing means the caller builds
`WebAssembly.Memory`, copies the segments in, and the module addresses that same buffer. The import appears only when the
body touches memory, so a function that reads nothing still needs nothing.

A register-relative operand is the same idea with the arithmetic moved to run time, because a pointer is a register
holding a virtual address. `0x400100(%rcx)` becomes "subtract the image base from `rcx`, add the displacement", and
`(%rcx,%rax,4)` adds the index shifted left by two. Both are bound-checked before the access: WASM has one flat buffer and
no faulting pages, so an address outside the image would read whatever sits at that offset and look entirely legitimate.
The check is unsigned against `limit - width`, which catches a null pointer (a small negative offset becomes a huge unsigned
one), an address past the end, and an access whose last bytes run off the end, all with the same comparison, and it traps
rather than reading. The bound is the memory the module imports, so it is the caller's memory and not a number invented
here.

Everything else is refused by name and offset: a segment override such as `%fs:` (WASM has no segments, and reading `0x28`
as a plain address would silently load the wrong memory), a constant address no segment maps, `syscall` when the layer is
off, `setcc` and the flags it needs, `jo` and its relatives, a `ret` with a stack adjustment, a block that does not return,
and code after `ret`.
Refusing is the honest option while the rest of the machine does not exist: a translator that skipped an instruction would
produce a module that runs and computes the wrong number.

### The stack

A function that uses the stack says so with `stackPages`, and the prologue points `%rsp` at the top of the imported memory:
the image occupies the bottom pages, the stack grows down from the top, and its floor is the image's last page. That floor
is checked on every `push` and on `pop`/`leave`, so a descent into the image traps rather than overwriting the program and
producing a wrong answer two instructions later. `push` is a descent followed by a store, `pop` a load followed by a rise,
and `leave` is `mov %rbp, %rsp` then `pop %rbp`, which is what makes a frame re-entrant: the bytes stay where they are and
the accounting forgets them.

`stackPages` is opt-in rather than a default because it changes what the caller must allocate: the import minimum grows, so
a caller who supplies the image's pages alone gets a refusal at translation and an instantiation error at load, rather than
a program with a stack it did not size.

### Calling another function

The registers are module globals rather than locals, and that is the calling convention rather than a shortcut around one: a
register has to outlive a call, because a callee that pushes `%rbp` must push the caller's `%rbp`. Arguments are already where
SysV puts them when the `call` executes, so passing one is just writing `%rdi`; `call` is a WASM `call`, `ret` is a WASM
`return`, and the result comes back in `%rax`, which is the whole of the convention this claims.

A callee is found by following the call's displacement into the code region the caller supplied, so `translateProgram` is the
entry point whenever anything is called. Its extent is settled by reachability from its first byte, not by cutting at the
first `ret`: a compiled function usually has several returns and one shared epilogue, and the bytes after the first `ret` are
still part of it. Only instructions the walk actually reaches are decoded, which matters because the bytes after a function
are the next function or the linker's zero padding, and a strict decode of those would fail on bytes that were never
instructions. What the walk leaves unreachable inside the range is refused: that means the address is not a function's start,
or a branch lands inside another instruction.

A call whose target is outside the region is refused by name. That is where libc lives, and reaching it needs more than a
calling convention: libc's own code calls into a partly relocated image, so it is refused rather than translated into
something that reads the wrong address. The kernel is reachable the way a program reaches it anyway, through `syscall`.

`lea` is the one memory operand that is *not* rebased. A register here holds a virtual address, and a load through it
subtracts the image base, so `lea message(%rip), %rsi` must leave `%rsi` holding `message`'s virtual address rather than a
buffer offset: rebasing it there would address the wrong byte, and the load that follows would fault by exactly the base.
For the same reason `lea` gets no bound check. An address one past the end of memory is how a compiler computes the end of
an array, and refusing it would refuse a program that never reads the result; a load from the same address still traps.

### The syscall layer

`syscall` is a call to an imported host function taking the call number and six arguments. Nothing is moved to make that
call: `%rax` holds the number and `%rdi`, `%rsi`, `%rdx`, `%r10`, `%r8`, `%r9` hold the arguments, which is where the
instruction itself puts them, so the shim reads the registers the kernel reads. `%rcx` and `%r11` are clobbered, as the
real instruction clobbers them, and flags are not trusted afterwards.

Four calls are implemented, and that is the whole claim: `write`, `exit`, `exit_group`, and `arch_prctl` for the thread
pointer. Everything else returns
`-ENOSYS`, which is what a kernel returns for a call it does not have, so a program that needs one fails the way it would
on a machine without it rather than silently reading zeroes. A `write` whose buffer is not entirely inside the image returns
`-EFAULT` instead of copying whatever follows it. An `exit` does not return, so it throws out of the module and the machine
records the status: a WASM function cannot be stopped from inside an import any other way, and returning normally would let
the program keep running past its own end.

The memory is created by the machine, not by the module, because the kernel reads it too: `write` copies bytes out of the
same buffer the program wrote them into. `syscalls: true` is opt-in for the reason `stackPages` is: it adds an import, so a
caller who does not know about it gets a refusal at translation rather than an instantiation error at load.

An imported function occupies index 0 of the WASM function index space, which shifts every translated function up by one.
That is the kind of arithmetic that produces a module that validates and calls the wrong function, so it is pinned by a test
that calls a callee in a module that also has the import, and that ends with the callee's own exit status.

### The thread pointer

`%fs:` addressing is how a compiler reaches thread-local storage, and the one place every libc-linked binary goes first:
`mov %fs:0x28, %rax` in a function prologue is the stack canary's read. A `%fs:` operand becomes the thread pointer plus its
displacement, with a base register as an offset inside the block and the image's base subtracted once at the end, so an
address in thread-local storage goes through the same bound check as every other load.

The thread pointer is a mutable `i64` global the module declares after the register file and *exports*, and the host writes
it through the exports object when `arch_prctl(ARCH_SET_FS)` runs. Exported rather than imported, because an imported global
takes index 0 of the global index space and would renumber all sixteen registers: a module with that mistake still loads,
and every register is quietly one slot off. The four prose registers keep 0 to 15 either way.

`arch_prctl` answers `ARCH_SET_FS` and `ARCH_GET_FS`, `-EINVAL` for an operation it does not have, and `-EFAULT` for a
`GET_FS` whose result slot is not mapped. `%gs` and the other segment overrides are still refused by name, because their bases
are not kept.

Two things the processor decided rather than this file. `lea %fs:0x10, %rax` loads `0x10`: the segment override is
ineffectual on `lea`, binutils warns about it, and the native run confirms it, so `lea` from `%fs:` is refused with that
reason instead of translated into a disagreement with the hardware. And a stack-protected function compiled by gcc ends with
`call __stack_chk_fail` followed by the linker's zero padding, so a `call` whose successor cannot be decoded now ends the
function's extent: a returning call must have an instruction after it, and padding is not one. Off the end of the bytes is
still truncation, and still refused.

`tests/thread.test.ts` is the evidence, and it is a differential: six programs compared with Linux's answer. A canary written
and read through `%fs:0x28`; the base moved twice mid-program, which is what proves the pointer is read at run time rather
than resolved at translation; a base register used as an offset inside the block; the `GET_FS` round trip; the two error
codes; and a trap when `%fs` was never pointed anywhere, where the bound check is the only thing between a thread-local read
and a read of the image's own first bytes. Then a function compiled by `gcc -fstack-protector-strong` with a real frame,
whose answer agrees with the processor's.

### The loader

`loadExecutable` is the whole of a load in one call: choose a base, place the image, copy the segments, then write the
relocations. The order is the part that matters. Relocations overwrite bytes the file supplied, so they go after the copy; and
they go before anything executes, because a program that starts with a half-fixed-up image is a program whose first indirect
reference goes wherever the zeroes point.

A position-independent executable is linked at address zero and holds offsets where a non-PIE executable holds addresses, so
the load base is a decision rather than a detail. It defaults to `0x400000`, and `0` is refused by name: loading a PIE where it
says it wants to be would map the null page, so a translated program that dereferences a null pointer would read the ELF
header and keep going instead of trapping, and every "is this pointer valid" check downstream would answer yes. Moving the
image shifts every segment by the same amount, which leaves each segment's offset in the buffer, the entry's offset, and the
page count exactly where they were; the amount is kept as `loadBias`, because that is what the relocation table is measured
from.

One relocation type is applied: `R_X86_64_RELATIVE`, whose slot becomes `loadBias + addend`. Both halves of the entry are in
the file's address space, not the loaded one, so the slot is written at `loadBias + r_offset` holding `loadBias + r_addend`.
Reading the offset as a load-time address is the mistake that puts every write `0x400000` bytes below where it belongs, which
lands inside some other segment often enough to look like a wrong value rather than a missed rebase; there is a test that
rebases a slot to the last eight bytes of a segment for exactly that reason.

Everything else is refused by name: `IRELATIVE`, which names a resolver the loader would have to *call* before the program
starts; `GLOB_DAT` and `JUMP_SLOT`, which need a symbol a dynamic linker would resolve; `COPY`; and the symbol-address forms.
A slot outside every mapped segment, a slot whose eight bytes run off the end of its segment, and a slot in a segment the file
marks read-only are each refused rather than written: a loader that wrote into a read-only page would produce a program whose
constants had been quietly overwritten by its own relocations, and the flag is the check that the mapping is the one the
linker had in mind.

`Machine` takes the memory to run in rather than always building one, and that seam is what makes any of this observable: the
image the loader produced, with its relocations written into it, is the buffer the program reads. A machine that allocated its
own would run a copy taken before any relocation was applied, and the differential below would report 1 in both directions for
a reason that has nothing to do with the loader.

The evidence is a differential, not a reference number. `tests/loader.test.ts` links a small PIE by hand whose `.data` holds a
relocated address of `_start`, and whose `_start` compares that pointer with its own runtime address: it exits 0 if they match
and 1 if they do not. The same bytes, translated and run twice, exit 0 with the relocation applied and 1 without it. That
second run is the one that makes the first meaningful, because a test that only ever sees 0 would pass on a program that
compares nothing.

A real libc static PIE is where the subset shows its edge: this loader applies all 1,305 of its relative relocations, checked
slot by slot against `readelf`'s own table, and then refuses the 24 `IRELATIVE` entries that follow. So the image is
relocatable and still not runnable, and the reason is not relocation: ifuncs have to be *called*, and a `static-pie` glibc
binary also wants `arch_prctl` for its thread pointer. Both are named next steps rather than hidden limitations.

## Checked against the processor, not against itself

`tests/native-agreement.test.ts` is the one check here that does not compare the translator with something this repository
wrote. Each program in `tools/native/` is assembled by binutils, the exact bytes `objcopy` extracts are decoded and
translated, and both versions are run: natively through a four-instruction `_start` that exits with the function's own
result, and as a WASM module. The numbers must agree.

Currently eight programs, and they agree: a countdown loop (5), a sum loop (10), one that leaves from the middle of a loop
(10), nested loops (6), the if/else shape (10), and three that turn on the difference between signed and unsigned
comparisons at each width (3, 2 and 3). It skips rather than fails when binutils is absent, because a cold checkout must
run with no toolchain.

`tests/syscalls.test.ts` closes the loop the roadmap was aiming at: `hello world` is written in assembly, built by `as` and
`ld`, run by Linux, and then its own `_start` bytes are translated and run against the shim. The executable's program headers
become the image, so `lea message(%rip)` resolves against the address the loader would have chosen, and the two answers must
match: `Hello, world!` on stdout, and a status.

`tests/memory-agreement.test.ts` does the same for memory, and does it end to end. Each program in
`tools/native/memory/` is assembled and linked at whatever addresses the linker picks, the resulting *executable's* program
headers are read into a `MemoryImage`, its `PT_LOAD` segments are copied into a fresh `WebAssembly.Memory`, the bytes the
symbol table gives for `target` are translated with that image as its address space, and both versions run. Two programs:
an absolute store and load (42), a pointer in a register that is read through, added to, written back, and read again with
a displacement (12), a stack frame with a spill, a push/pop pair and a `leave` (10), and a call across two functions (13). Each one's expected number is stated in the test rather than taken from either side, and the
store is then read out of the caller's buffer at the offset the image maps for the symbol. The linked binary's own headers
choose every address, so no part of that test hardcodes one.

## Roadmap

1. **ELF loading** (done): ELF64 headers, `PT_LOAD` segments mapped into a WASM `MemoryImage`, `RELA` relocations parsed and
   `R_X86_64_RELATIVE` applied against a chosen load base, and explicit refusals for 32-bit, big-endian and relocatable
   objects.
2. **Decoder** (done for the hello-world subset): ModRM, SIB, REX, RIP-relative addressing, and named refusals. SSE,
   `x87`, the string instructions and 16-bit operands are outside it, and an `objdump`-driven fixture keeps the boundaries
   honest.
3. **Translator** (in progress): arithmetic, flags as comparisons, forward branches, loops, memory against the mapped image
   including register-relative and indexed addressing with a bound check, the stack with a checked floor, and calls between
   translated functions with the registers as module globals, and `syscall`, results checked by running the module and,
   where the toolchain allows, by the processor.
4. **Syscall layer** (started): `write`, `exit` and `exit_group` answer; everything else is `ENOSYS`. Next are `brk`, `mmap`
   and the file syscalls, each behind a capability check.
5. **Evidence** (started): `hello world` runs under real Linux and under the translator, with stdout and exit status compared.
   Next is a binary whose libc calls need relocations, which is a loader problem before it is a translator one.

## Non-goals

- Not a full-PC emulator. No BIOS, no devices, no boot.
- Not a JIT for arbitrary self-modifying code initially; self-modifying code lands with explicit invalidation or not at
  all.
- No `fork`/`execve`, so no shells and no `npm install` *by this route*.

## Licence

MIT.
