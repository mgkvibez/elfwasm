# The programs the processor checks

Each `.s` file here is a function named `target`, ending in `ret`, written in the subset the translator covers. They are
not fixtures for the test suite to read as bytes: they are input for the assembler, and the test takes whatever bytes
`as` emits and `objcopy` extracts.

`tests/native-agreement.test.ts` links each one against a four-instruction `_start` that exits with the function's result,
runs it natively, then decodes and translates the same bytes to a WASM module and runs that too. Native exit status and
the module's return value must agree, and the value is what the program computes rather than what a test decided it
should be.

The test skips when `as`, `objcopy` and `ld` are missing, because a cold checkout has to run `npm test` with no toolchain
at all. Where the toolchain exists, it runs and it is the strongest check in this repository: every other test compares
the translator to a number written by hand.

Adding a program: write the `.s` file, end it in `ret`, keep the result in `rax`, and make sure the result fits in the
low eight bits, since that is what a process exit status carries.

## rotate.s

The two rotates, at every width the translator reads them: the swap-halves idiom (`ror $32`), a byte rotating inside its own eight
bits, a count equal to the width which is no rotation at all, a word rotated through `%cl`, and the two flag cases a branch depends
on, a comparison that a rotate must not disturb. Every case compares against a constant that can be read off the definition, and a
failure returns the number of the case, so the exit status names what broke rather than needing a second run.

All the cases share one failure label, which is not only tidy: a jump out of the block it sits in is refused now (see the README's
note on that shape), and a program whose every branch lands in one block after the last check never asks for it.

## The two shuffles, which are not here

`shufps` and `shufpd` have no program in this directory, and the reason is the host rather than the instructions. The sandbox this
was developed in runs under gVisor, and its `shufps` is not the architecture's: with the destination `[1,2,3,4]` and the source
`[5,6,7,8]`, the immediate `0x1b` returned `[4,3,6,5]` and `0xe4` returned lanes 2 and 3 from the source's *high* dwords. Measured
with `movq` and `punpckhqdq`, so the reading does not depend on `shufps` itself, the two answers contradict each other: no single
mapping of selector to lane gives both. Its `shufpd` matched the manual on all four immediates, so the fault looks like `shufps`
alone.

A program here would therefore enshrine the emulator's answer, or fail on this host while passing elsewhere. The expectations live in
`tests/shuffle.test.ts` instead, written from `_mm_shuffle_ps` and `_mm_shuffle_pd` and named per selector so a change in the mapping
fails a specific line. If this ever runs on a processor rather than under gVisor, a program here would be worth adding back.
