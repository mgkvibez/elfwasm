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
relocations), `%gs` and every segment base but `%fs`, `munmap` and `mprotect`, file-backed `mmap`, a heap that cannot grow
past the pages the machine was created with, and every syscall but `write`, `exit`, `exit_group`, `arch_prctl`, `brk`
and `mmap`. Two honest limits of the mapping:
WASM linear memory has no per-page protection, so the `read`/`write`/`execute` flags are carried for the translator to
honour rather than enforced by hardware; and the 32-bit address space means an image larger than 4 GiB is refused.

**Tests:** 265

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

Six calls are implemented, and that is the whole claim: `write`, `exit`, `exit_group`, `arch_prctl` for the thread pointer,
and `brk` and `mmap` for allocation. Everything else returns
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

### Bytes, and the registers that are inside other registers

An 8-bit register is not a variable of its own, it is a byte of a 64-bit one. `%al` is bits 0 to 7 of `%rax`, and `%ah` is bits
8 to 15 of the same register, which is why `mov $0xff, %dh` must leave `%dl` alone and `mov %dh, %al` must leave `%rdx` alone.
The translator keeps the 64-bit local and reads and writes the byte through it: a read shifts and masks, and a write reads the
bits around the target back, masks them, and ors the new byte in. There is no second local for a byte, so the two names cannot
disagree.

An 8-bit comparison is evaluated at 64 bits on sign-extended bytes. Without that, a byte holding 0x80 compares as 128 rather
than as -128 and every signed branch after a byte comparison goes the wrong way. `cmp %dh, %dl` followed by `jl` reads bit 7 of
a byte, and it has to mean that.

The multi-byte nop, `0x0f 0x1f`, is the alignment padding a compiler emits between functions, and it takes a ModRM whose only
defined register field is `/0`; the four `0x0f 0x18` prefetch hints are addresses to read into a cache and nothing else. Both
decode and both translate to nothing at all: no register, no flag, and no memory read even when the operand names an address,
which is why a nop is translated by a function whose image was never supplied. The linker's own padding spells the nop with a
`0x66` prefix and was handled earlier; the bare form is what a compiler emits.

A condition is machine state, not a property of one straight-line run. `cmp` and `test` store their two operands in locals, and
every flag-writing instruction stores the pair `cmp` would have stored, so a condition stays readable as far as the program is
concerned for as long as nothing overwrites it. Three consequences, and each one was a refusal before it was a rule. `lea`
computes an address and writes no flag, and neither do `push` and `pop`, so none of them hides a comparison made before them.
Both arms of an if/else enter with the flags as the branch left them, because a `jcc` changes no flag, and the join afterwards
holds them only if the two arms end holding the same width. `cmov` reads them from wherever they were written, which is what one
sits between a comparison and a branch for.

A conditional move, `0x0f 0x40` to `0x0f 0x4f`, is the other instruction a resolver reaches once it has compared: the
destination keeps its value unless the condition holds, in which case the source is written into it. It is translated as an `if`
around a write rather than as a branch around a `mov`, because it leaves the flags exactly as it found them, which is the whole
reason to use it between a comparison and the branch that reads the same flags. The width comes from the destination, and a
32-bit answer zero-extends, so a `cmove %ebx, %eax` clears the upper half of `%rax` the way the processor does. The overflow
and parity conditions are refused by name, as they are for `jcc`, because neither flag is computed.

The byte forms have their own opcodes, and the subset now has them: group 3 (`test`, `not`, `neg` at `0xf6`), group 1
(`0x80`), the register-to-register pairs (`0x00`, `0x08`, `0x20`, `0x28`, `0x30`, `0x38`, `0x84`, `0x88` and their reverse
directions), the accumulator forms (`0x04`, `0x0c`, `0x24`, `0x2c`, `0x34`, `0x3c`, `0xa8`), and `mov r8, imm8` at `0xb0` to
`0xb7`, where codes 4 to 7 are the high bytes without REX and `spl` to `dil` with one.

Two facts about functions came out of the same work, and both are refusals rather than translations.

A call whose target is inside the body being translated, and is not one of that body's instruction starts, is refused: the
decoder will decode from the middle of an instruction and the bits it produces are a program that runs and computes something
the source never said. Decoding is not the same as being a function's start.

A call followed by a zero byte does not return. The linker pads with zeros, and a zero byte cannot start an instruction a
compiler would emit, so the call is to something that never returns, which is the shape `call __stack_chk_fail` has at the end
of a stack-protected function. This is tested *before* decodability, because a zero byte does decode (`add %al,(%rax)`) and a
rule that leaned on the decode failing would depend on which instructions the subset happened to have.

### The code that runs before the program

A `R_X86_64_IRELATIVE` relocation does not name a value, it names a *function*. The addend is the address of an ifunc resolver,
the loader calls it, and the address it returns is what goes in the slot. That is how glibc picks a `memcpy` for the machine it
finds itself on, and it is the last thing a static glibc PIE needs before its own `_start` can run.

So the loader takes a `resolve` callback rather than knowing how to run code: an `IRELATIVE` entry hands it
`loadBias + addend`, and whatever comes back is written into the slot *unbiased*, because it is already a load-time address.
Without the callback the relocation is refused by name, since a loader that cannot call the function has nothing to write and
writing the resolver's address would be a lie the program only discovers when it calls it.

`loadResolved` in `src/startup.ts` is that callback, implemented with the translator. Two things make running a resolver
different from running a program. There is no symbol table entry to give its length, and what a resolver does have is a `ret`,
so the extent is found by decoding from its address to the one it can *reach*, instruction by instruction: a `mov $0xc3, %eax`
holds that byte inside its immediate and a scan that stopped there would translate a third of a function, while a linear scan
would also walk straight past a jump into another function, because the first `ret` it reads is not the first one the resolver
can arrive at. So the walk follows the jumps it decodes, and the translator is handed the instructions rather than a byte slice.
And it is *called* rather than entered, which is `Machine.call`: the translator's module returns `%rax`, so calling the module and
taking its return value is calling the function.

Refused by name: a resolver with no `ret` within the walk limit, one that will not decode, one whose address is not in an
executable file-backed segment, and one whose `call` reaches outside its extent. Each is a limit of this layer rather than of the
instruction set, and each says so at the address it happened.

`tests/startup.test.ts` builds a PIE by hand whose `.data` holds an ifunc: the linker turns `slot: .quad resolver` into an
`IRELATIVE` entry, and the resolver returns a marker's address computed with `lea marker(%rip)`, which only it knows. The slot
then holds exactly that, at two different load bases, so the answer cannot be a link-time constant that happened to match. The
same bytes run natively **fail** with status 1: this binary is linked with `--no-dynamic-linker`, so nothing resolves its
ifuncs and the kernel jumps to the entry with the slot still holding the resolver's address. The differential is the two
statuses, and it is the wrong way round from every other test here on purpose: 1 without the relocations applied, 0 with them.

A real glibc static PIE **loads**. Every ifunc resolver it carries is translated, called, and its answer written through the
relocation: 1329 relocations applied on a `gcc -static-pie` hello-world, through eleven resolvers that between them test
`_dl_x86_cpu_features` for AVX-512, AVX, ERMS, RTM and SSE, and answer with the address of the routine this machine should use.

What that took is the last structural piece, and it is not an instruction: a resolver's control flow is a *graph*, not a tree. It
shares a tail between two paths, jumps back into a block that is also reached by falling into it, and exits from seven `ret`s. No
nesting of WASM blocks expresses that. So a body whose branches are not a tree is translated as numbered blocks with the block
number in a local, dispatched from a WASM `loop`: a branch sets the number and returns to the top, and the top runs whichever block
the number names. Registers are globals and a comparison's operands are locals, so a block carries no state of its own and one
copy of each block is enough. Dispatch is a linear scan of the block numbers, which is correct rather than fast.

It does not *run* yet: nothing calls `main`, and the entry point is the frontier.

### The entry point

`_start` in a static PIE is twelve instructions: `xor %ebp,%ebp`, `mov %rdx,%r9`, `pop %rsi`, `mov %rsp,%rdx`, `and $-16,%rsp`,
two pushes, two zeroing xors, `lea main(%rip),%rdi`, the `addr32 call` into `__libc_start_main`, and `hlt`. Three of those needed
something new, and none of them was arithmetic:

1. *`67 e8`, the 32-bit address size prefix in front of a call.* A call's displacement is not an address, so the prefix changes
   nothing about a call or a jump and is dropped there. Before a memory operand it is exactly the case it changes, and that is
   refused by name rather than reading a 64-bit address while the instruction asked for a 32-bit one.
2. *`hlt`, which ends an entry point the way `ret` ends a function.* It is decoded as its own instruction and translated to
   `unreachable`, because in user mode it traps: that is the trap, and it also says the bytes after it are dead. The extent walk
   stops at it, so an entry point needs no `ret` for its extent to be knowable.
3. *`cpuid`, which a program uses to ask what the machine can do.* This machine's answer is zero for every leaf: no vendor is
   reported and nothing beyond the base leaf is offered, so a program that asks takes the path for an x86-64 without any optional
   feature. That is a real machine, and inventing feature bits would send a program down a path this translator cannot run.

The twelve instructions all translate now, so the frontier is inside libc rather than in the entry itself. Translating the `call`
means finding the callee's extent, and walking `__libc_start_main` gets 0x152 bytes further than it did, to `+0x32e`.

### The vector register file

That 0x152 bytes needed SSE, because glibc packs two 32-bit arguments into one vector register and stores the pair. A vector
register is 128 bits and a WASM global is 64, so `%xmm0` to `%xmm15` are two `i64` globals each, a low half and a high half, and
they sit after the general file in the global section. Naming a half is what makes the rest cheap: the low half of a vector
register is a register that holds 64 bits, so `movq %xmm0, 0x20(%rsp)` is the store it looks like and goes through the existing
`mov`-to-memory path with no new address arithmetic.

Four encodings are in the subset, and they are the ones a startup uses:

1. `66 0f 6e`, `movd` into a vector register: 32 bits, or 64 with REX.W, from a general register or memory. The whole 128 bits are
   written, because the upper half is cleared, which is a second write of zero rather than a special case.
2. `66 0f 7e`, the same in the other direction: four bytes into a general register or into memory.
3. `66 0f 62`, `punpckldq`: the four low dwords of the two operands, interleaved, so each half of the destination is a dword of its
   own low half or'ed with a dword of the source's, shifted up. Both halves are read into scratch locals first, because the
   destination's low half is written after its high half has been computed from it.
4. `66 0f d6`, `movq` out to memory: the low 64 bits of a vector register.

`f3 0f 7e` is the load that mirrors it: `movq` into a vector register from memory or from another one, with the upper half cleared.
The same two opcode bytes under `0x66` are the store, so the prefix is what decides which direction is meant.

### The accumulator instructions

`idiv`, `div`, `mul` and `imul` in their one-operand forms read and write registers the encoding does not name: `%rax` is the
accumulator and `%rdx` is the high half of a product or the remainder, so `%rdx:%rax` is a 128-bit dividend or a 128-bit product.
WASM has neither a widening multiply nor a 128-bit divide, so both are built out of 64-bit arithmetic:

1. *The widening multiply* is the four partial products of the two 32-bit halves of each operand, with the middle column summed
   out of both halves' carries, which is where the product's high 64 bits come from.
2. *The division* is long division, a bit at a time, 128 times: shift the remainder up, bring in the dividend's next bit, and subtract
   the divisor when it fits. The trial subtraction is `rem - divisor * take` rather than a pick between two remainders, because the
   condition is an `i32` and the operands are `i64`, and multiplying by a 0 or a 1 is the same choice without a conversion.
3. *The processor's `#DE`* is a trap here too: a zero divisor, and a quotient that would not fit in `%rax` (which is the check that
   the dividend's high half is smaller than the divisor). `idiv` works on magnitudes, negates the quotient when the signs differed
   and the remainder when the dividend was negative, since the processor rounds toward zero.
4. *`cwde`, `cdqe`, `cdq` and `cqo`* now translate, because `cqo` is how a compiler makes `%rdx:%rax` one signed number before
   dividing it. Both halves of the widening are a shift out and back: `%rdx` is `%rax` shifted down arithmetically by 63.

### Half-width and whole-register moves

The four half-width moves name one half of a vector register and leave the other one alone, which is why they exist: a routine that
has packed one half fills the other with one of them. `0f 12` and `0f 13` are the low half and `0f 16` and `0f 17` the high one, with
`0x12`/`0x16` loading and `0x13`/`0x17` storing. Their register forms are shuffles between two vector registers rather than moves, so
those are refused by name.

The whole-register moves are `0f 28`/`0f 29` (`movaps`), `0f 10`/`0f 11` (`movups`), `66 0f 6f`/`66 0f 7f` (`movdqa`) and
`f3 0f 6f`/`f3 0f 7f` (`movdqu`), which differ only in an alignment promise the emulated memory has no way to break. WASM has no
16-byte load or store, so each is two 64-bit accesses at the same address, eight bytes apart: the low half at the operand's own
offset and the high half at the memarg's next one. Their register forms are refused for the same reason the shuffles are.

The four integer logical operations are in as well: `pand`, `pandn`, `por` and `pxor`, each bitwise, each applied to both halves
independently. `pandn` is the one that is not symmetric, computing `~destination & source`.

### A call whose target is a value

`call *target` reads its target from a register or from memory, so which function runs is decided at run time and cannot be an index
in the module's function list. What it can be is one of the functions this program translated: the target is compared against each of
them and the one that matches is called. A target that matches nothing is `unreachable`, which is the trap a jump to garbage is.

The set of candidates is the functions the translation knows, which is the entry plus everything a direct call reached. A function
whose address reaches the program only as a pointer is in it only when the caller says so through `indirectTargets`, and that is not a
detail of the implementation but a property of the question: `main`'s address is passed to `__libc_start_main` as an argument, so no
call in the bytes names it.

That makes the bodies of a program something to emit twice. An indirect call is a comparison against every function, so the list has to
be complete before any body is written to, and translating the bodies is what completes it: the first pass grows the list and is thrown
away, the second emits the same bodies against a list that cannot grow further, because the same bytes make the same decisions.

`sbb` comes with it, because `sbb %rbx,%rbx` is how a compiler turns a comparison into a mask: a register minus itself is zero, and
zero minus the borrow is all ones or zero. The borrow is read out of the comparison the flags still hold, since `a - b` borrows exactly
when `a < b` as unsigned values, and a `test` borrows nothing at all. Every other flag `sbb` sets is dropped rather than computed, so
what follows it knows nothing about what it left behind.

### A jump whose target is a value

A computed jump lands in a block of the same body, so it is a block number to choose rather than an address to call. The blocks of a
body are already numbered for the dispatcher, so the target is compared against the address each block starts at, the match sets the
block number, and the dispatcher runs it. A target that is not where a block starts is `unreachable`.

### Alignment padding

An assembler pads a block out to a boundary so the block after it starts where the processor likes to fetch, and nothing ever jumps to
the padding itself: `nopl 0x0(%rax,%rax,1)` sits between two blocks and is reached by no path. An extent that insisted every byte
inside a function is reached refused a function the assembler wrote, and the walk now carries a `nop` that nothing reaches as the dead
code it is. Any other unreachable byte is still refused, because it may be data or the middle of another instruction, and a `nop` is
the only padding this decoder knows.

### The jump table a switch reads

A `switch` over a dense range of values does not name its cases in its bytes: gcc emits four instructions that read a table of 32-bit
offsets, and each entry is where a case starts.

```
lea    table(%rip),%rcx       the table's own address, which is what its entries are measured from
movsxd (%rcx,%rdx,4),%rdx     one entry, sign-extended: an offset rather than an address
add    %rcx,%rdx              the offset plus the table, which is where the case starts
jmp    *%rdx
```

That shape is recognised, the table is read out of the image's file bytes, and every entry becomes a place the extent walk continues
from and a block the dispatcher can land in. How many entries there are is not in the table, so the count comes from the bounds check
the compiler put before it (`cmp $0x22,%edx; ja default` is twenty-three entries). Without that comparison no table is read at all: a
table read past its end names a target in some other function, and an early version of this walked into unrelated code exactly that way.

Two things the switch needed besides the table. `movsxd` translates now, because a table entry is a signed offset. And a body may end in
`hlt`: an entry point that calls `exit` and halts leaves its answer in `%rax` where a call's would have gone.

RIP-relative operands are now resolved against the body's own address rather than the program's. They agree for the entry point and differ
for every function it calls, and reading a callee's globals against the entry's neighbourhood was wrong.

### 16-bit operands, and the 0x66 prefix

`0x66` is not a separate instruction set, it is a width: `66 89 15 4e 3d 07 00` is `mov %dx,0x4b112c`, the same store as the 32-bit
form with a two-byte value and a two-byte register name. The prefix records a 16-bit operand size and the decode continues, with REX.W
winning over it (`66 48 8b` is a 64-bit operand with a prefix in front of it).

Two things follow from the width rather than from the prefix.

The immediate shrinks with it: `66 b8 34 12` is two bytes of immediate, and the `81`/`c7` group reads imm16 rather than imm32. It
shrinks only for those: `48 c7 c0 01 00 00 00` is `mov $1,%rax` with four bytes of immediate, sign-extended, and reading eight would
take three bytes of the next instruction with it. That mistake desynchronised the decode of four of the assembler fixtures at once,
which is how it was found.

And a 16-bit register is a half of a 64-bit one, not a variable of its own. `%ax` and `%r8w` are the low 16 bits of `%rax` and `%r8`,
read through a `0xffff` mask and written by merging, so the upper 48 bits survive a narrow move. The translator's `mov` family now takes
that width in all three shapes: register to register, register to memory, and memory into a register, where the load zero-extends like
every other narrow load. 16-bit arithmetic is still refused by name, because its flags are 16 bits wide and this translator computes them
at 64.

Where it stops now: offset `+0x2e1` into the same function, on `66 0f 6c c1`, which is `punpcklqdq`: an SSE shuffle that interleaves the
low halves of two vector registers. The 16-bit store that stopped the walk at `+0xa7` translates, and what stops it next is a vector
instruction again rather than a width.

The cost of the dispatcher is worth stating: a body with any backward branch takes that path, so even a clean loop is expressed by
naming blocks rather than as a WASM `loop`. The structured path is kept for the bodies that are trees, which is most of the small
programs here.

### The heap

The heap starts where the image's own pages end. That placement is the whole design: a virtual address and a WASM offset stay
one addition apart from the image's base all the way to the top of the memory, so a heap address needs no second mapping table
and `write` reads it with the same arithmetic it reads the image with. `brk` grows upward from there and `mmap` grows downward
from the top of the free region, which is the direction Linux grows its own mappings, and neither may cross the other. `brk`
answers with the break's new value on success and its old one on failure, as Linux does, which is how a caller tells them apart.
A `brk` below the image, past the memory, or into a mapping is a failure that moves nothing.

The stack is the caller's to declare. `%rsp` starts at the top of the memory, so a break that grew that far would hand a program
the pages its own frame is on: `new Machine({ image, pages, stackPages })` says how much of the top is the stack, and the heap
stops short of it. Without `stackPages` the heap may use the whole memory, which is right for a program translated without a
stack and wrong for one with a frame.

`mmap` serves one kind of mapping: private, anonymous, page-rounded, zero-filled. A file mapping is refused with `-ENOSYS` rather
than served as anonymous memory, because a program that maps a file and finds zeroes has a working address containing the wrong
bytes, which is worse than a failure; and `MAP_FIXED` is refused for the same reason, since the address it names may be the
image. A zero length is `-EINVAL` and a mapping that does not fit is `-ENOMEM`. `munmap` and `mprotect` are not implemented and
return `-ENOSYS`: WASM has no page protections to change, and claiming a mapping was released while its bytes stay readable is
a claim this layer cannot keep.

What it hands out is zeroed explicitly, because a new `WebAssembly.Memory` starts zeroed and hides the bug on the first run: the
kernel zero-fills the pages it gives away, and a range that was written before it was given back is not zero by accident.

`tests/heap.test.ts` is the evidence. Unit tests pin the initial break, growth, refusal at the limit, the stack reservation, and
the zero-fill across a shrink and a grow; then `mmap`'s alignment, its page rounding, the two refusals, and the collision where
mappings reach the break. The end-to-end test is a program that asks `brk` for a page, copies eight bytes off the image into it,
and writes six of them to stdout: the address comes back from the kernel, the program writes through it, and `write` reads it
back out of the same memory. Linux prints `heap!` and exits 0, and so does the translated run.

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
