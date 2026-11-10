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
