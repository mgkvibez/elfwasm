/**
 * The calling convention: arguments in registers, `call` as a WASM call, `ret` as a return.
 *
 * The registers are module globals, so a call passes its arguments the way SysV says to: by writing `%rdi`, `%rsi`, `%rdx`
 * and the rest. That is not a shortcut around a calling convention, it is the calling convention: a register outlives a
 * call, and a callee that pushes `%rbp` has to push the caller's.
 *
 * Every program here is written as raw bytes at a fixed address, because the point of each test is the shape of the code a
 * compiler emits, not the assembler's choice of encodings.
 */

import { describe, expect, it } from "vitest";

import { translateProgram } from "../src/translate.js";

const runtime = (
  globalThis as unknown as {
    WebAssembly: {
      Memory: new (descriptor: { initial: number }) => { readonly buffer: ArrayBuffer };
      Module: new (bytes: Uint8Array) => unknown;
      Instance: new (
        module: unknown,
        imports?: { env: { memory: { readonly buffer: ArrayBuffer } } },
      ) => { exports: Record<string, () => bigint> };
    };
  }
).WebAssembly;

const ORIGIN = 0x401000n;

/** One page of address space at the origin, so a stack can sit above it in the test that needs one. */
const EMPTY_IMAGE = {
  baseAddress: ORIGIN,
  pages: 1,
  offsetOf: (virtualAddress: bigint) =>
    virtualAddress >= ORIGIN && virtualAddress < ORIGIN + 0x10000n ? Number(virtualAddress - ORIGIN) : undefined,
};

function hex(text: string): number[] {
  return text.trim().split(/\s+/).map((byte) => Number.parseInt(byte, 16));
}

/** Lay bytes out at a given offset in a code region, with `0x90` (nop) filling whatever is skipped. */
function region(parts: readonly { at: number; bytes: number[] }[], size: number): Uint8Array {
  const bytes = new Uint8Array(size).fill(0x90);
  for (const part of parts) bytes.set(part.bytes, part.at);
  return bytes;
}

function run(code: Uint8Array, entrySize: number, stackPages = 0): bigint {
  const module = translateProgram(code.subarray(0, entrySize), "main", {
    code: { bytes: code, virtualAddress: ORIGIN },
    codeAddress: ORIGIN,
    // one page of image plus the stack pages, which is where %rsp starts
    ...(stackPages === 0 ? {} : { stackPages, addresses: EMPTY_IMAGE, memoryPages: 1 + stackPages }),
  });
  if (stackPages === 0) return new runtime.Instance(new runtime.Module(module)).exports.main!();
  const memory = new runtime.Memory({ initial: 1 + stackPages });
  return new runtime.Instance(new runtime.Module(module), { env: { memory } }).exports.main!();
}

describe("one function calling another", () => {
  it("passes an argument in %rdi and reads the result out of %rax", () => {
    /*
     * entry at 0x401000:  mov $0x29, %rdi ; call 0x401020 ; ret
     * callee at 0x401020: mov %rdi, %rax ; add $1, %rax ; ret
     */
    const code = region(
      [
        { at: 0x00, bytes: hex("48 c7 c7 29 00 00 00 e8 14 00 00 00 c3") },
        { at: 0x20, bytes: hex("48 89 f8 48 83 c0 01 c3") },
      ],
      0x30,
    );
    expect(run(code, 0x0d)).toBe(0x2an);
  });

  it("follows a call inside a call, which is where a stack of frames would be", () => {
    /*
     * entry:  mov $5, %rdi ; call outer ; ret
     * outer:  call inner ; add $2, %rax ; ret
     * inner:  mov %rdi, %rax ; add $1, %rax ; ret
     */
    const code = region(
      [
        { at: 0x00, bytes: hex("48 c7 c7 05 00 00 00 e8 14 00 00 00 c3") },
        { at: 0x20, bytes: hex("e8 1b 00 00 00 48 83 c0 02 c3") },
        { at: 0x40, bytes: hex("48 89 f8 48 83 c0 01 c3") },
      ],
      0x50,
    );
    expect(run(code, 0x0d)).toBe(8n);
  });

  it("recurses, which is the same machinery one level deeper", () => {
    /*
     * countdown(n) = n == 0 ? 0 : countdown(n - 1) + 1, so the result is n. It has two rets and a shared epilogue, which is
     * the shape the extent walk has to find: a function is not the bytes up to its first ret.
     *
     * entry: mov $5, %rdi ; call countdown ; ret
     * countdown: test %rdi, %rdi ; je zero ; sub $1, %rdi ; call countdown ; add $1, %rax ; ret ; zero: mov $0, %rax ; ret
     */
    const code = region(
      [
        { at: 0x00, bytes: hex("48 c7 c7 05 00 00 00 e8 14 00 00 00 c3") },
        // the call is at 0x28, so the recursion's displacement is measured from 0x2d back to 0x20: -13
        { at: 0x20, bytes: hex("48 85 ff 74 0e 48 83 ef 01 e8 f2 ff ff ff 48 83 c0 01 c3 48 c7 c0 00 00 00 00 c3") },
      ],
      0x40,
    );
    expect(run(code, 0x0d)).toBe(5n);
  });

  it("keeps a callee's view of %rbp out of the caller's when the callee saves it", () => {
    /*
     * A frame is the caller's, so a callee that wants one pushes %rbp and pops it back. If the registers were per function
     * instead of globals, that push would save zero and the caller's %rbp would be replaced by whatever the callee chose.
     *
     * entry: mov $0x11, %rbp ; mov $3, %rdi ; call f ; mov %rbp, %rax ; ret
     * f:     push %rbp ; mov %rsp, %rbp ; mov %rdi, %rbp ; mov %rbp, %rax ; pop %rbp ; ret
     */
    const code = region(
      [
        { at: 0x00, bytes: hex("48 c7 c5 11 00 00 00 48 c7 c7 03 00 00 00 e8 0d 00 00 00 48 89 e8 c3") },
        { at: 0x20, bytes: hex("55 48 89 e5 48 89 fd 48 89 e8 5d c3") },
      ],
      0x40,
    );
    expect(run(code, 0x17, 1)).toBe(0x11n);
  });
});

describe("what a call will not do", () => {
  it("refuses a target outside the code region, and says why", () => {
    // call 0x900000, which is not in the region at all
    const code = region([{ at: 0x00, bytes: hex("e8 fb ef 4f 00 c3") }], 0x10);
    expect(() =>
      translateProgram(code.subarray(0, 6), "main", {
        code: { bytes: code, virtualAddress: ORIGIN },
        codeAddress: ORIGIN,
      }),
    ).toThrow(/outside the code region/);
  });

  it("calls a function that sits in a hole of a callee, which is another function's start and not the middle of this one", () => {
    /*
     * The callee jumps over a region and then calls into it, which is the shape a real binary has wherever a cold block lives a long
     * way from its function: the skipped bytes hold another function, and a call to one of them names a start rather than the middle
     * of anything.
     *
     * The walk used to report that target as "inside this function", because it lay below the far end of the reachable set, and in a
     * linked image the far end is most of the text. A body is now the instructions its entry reaches rather than the bytes between its
     * first and its last, so the skipped bytes belong to nobody and the call is a call.
     *
     * entry  at 0x00: call 0x20 ; ret
     * callee at 0x20: jmp 0x30 (over the region holding the function at 0x28) ; at 0x30: call 0x28 ; ret
     * at 0x28:        mov $0x2a,%rax ; ret, never fallen into, only called
     */
    const code = region(
      [
        { at: 0x00, bytes: hex("e8 1b 00 00 00 c3") },
        { at: 0x20, bytes: hex("e9 0b 00 00 00") },
        { at: 0x28, bytes: hex("48 c7 c0 2a 00 00 00 c3") },
        { at: 0x30, bytes: hex("e8 f3 ff ff ff c3") },
      ],
      0x40,
    );
    expect(run(code, 0x06)).toBe(0x2an);
  });

  it("refuses a call when no code region was given at all", () => {
    const code = region([{ at: 0x00, bytes: hex("e8 fb 0f 00 00 c3") }], 0x10);
    expect(() => translateProgram(code.subarray(0, 6), "main", { codeAddress: ORIGIN })).toThrow(
      /needs the code region/,
    );
  });

  it("refuses a target that is not where a function starts, rather than guessing", () => {
    /*
     * The entry calls into the middle of its own bytes. The reachability walk finds instructions the entry cannot reach, which
     * is the honest signal that the address is not a function's start: translating from there would produce code that loads,
     * runs, and computes something else.
     */
    const code = region(
      [
        // call 0x40100a, which is inside this function's own instruction stream
        { at: 0x00, bytes: hex("e8 05 00 00 00 48 c7 c0 01 00 00 00 c3") },
      ],
      0x20,
    );
    expect(() =>
      translateProgram(code.subarray(0, 0x0c), "main", {
        code: { bytes: code, virtualAddress: ORIGIN },
        codeAddress: ORIGIN,
      }),
    ).toThrow(/nothing reaches|is not an instruction|is not in the|is not an instruction start/);
  });
});

describe("what this machine says it can do", () => {
  it("answers xgetbv with x87 and SSE and nothing above them, so a program takes the path without AVX", () => {
    /*
     * `xor %ecx,%ecx ; xgetbv ; ret` asks for extended control register 0, which is where the operating system says which vector
     * state it enabled. The answer is 3: `x87` and SSE, no YMM state. Every AVX instruction sits behind the bit the answer leaves
     * clear, and none of them is decoded here, so the honest answer is the one that keeps a program on instructions that exist.
     */
    const code = region(
      [
        // xor %ecx,%ecx ; xgetbv ; ret, which answers in %rax
        { at: 0x00, bytes: hex("31 c9 0f 01 d0 c3") },
      ],
      0x20,
    );
    expect(run(code, 0x06)).toBe(3n);
  });
});

describe("a bit that becomes a flag", () => {
  /*
   * `bt value, index` writes the carry and leaves every other flag undefined. The carry is expressed the way a comparison already is,
   * as the unsigned-below pair in the two locals a `cmp` uses, so `jb` after a bit test means the bit was set.
   */
  it("sets the carry from the bit the index names, and jb follows it", () => {
    /*
     * mov $0x8,%eax ; mov $3,%ecx ; bt %ecx,%eax ; jb done ; mov $0,%eax ; ret ; done: mov $1,%eax ; ret
     *
     * Bit 3 of 8 is set, so the carry is set and jb is taken. The same bytes with the index at bit 2 ask about a zero bit.
     */
    const programme = (index: string): Uint8Array =>
      Uint8Array.from(hex(`b8 08 00 00 00 b9 ${index} 00 00 00 0f a3 c8 72 06 b8 00 00 00 00 c3 b8 01 00 00 00 c3`));
    expect(run(programme("03"), 0x1b)).toBe(1n);
    expect(run(programme("02"), 0x1b)).toBe(0n);
  });

  it("refuses a condition a bit test does not define, instead of answering it from the last comparison", () => {
    /*
     * `bt %rcx,%rax ; je done` asks whether the value was zero, which a bit test says nothing about. The answer is refused rather
     * than taken from whatever comparison happened to run before it.
     */
    // mov $8,%eax ; mov $3,%ecx ; bt %ecx,%eax ; je done ; ret ; done: ret
    const code = Uint8Array.from(hex("b8 08 00 00 00 b9 03 00 00 00 0f a3 c8 74 01 c3 c3"));
    expect(() => run(code, 0x12)).toThrow(/bit test|does not define/);
  });
});
