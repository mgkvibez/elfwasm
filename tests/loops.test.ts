/**
 * Loops: backward branches, become a WASM `loop` and a `br_if` back into it.
 *
 * The programs here are small enough to hand-assemble and check by reading, and each one's number is stated in the
 * architecture's terms rather than in the translator's. `tools/native/` holds the same shapes as assembler source, so the
 * processor itself checks them where binutils is installed.
 */

import { describe, expect, it } from "vitest";

import { translateBytes } from "../src/translate.js";

const hex = (text: string): Uint8Array =>
  Uint8Array.from(text.trim().split(/\s+/).map((pair) => Number.parseInt(pair, 16)));

/**
 * The runtime, reached through `globalThis` with the smallest surface this file needs.
 *
 * The project has no `dom` lib and no `@types/node` on purpose, so `WebAssembly` is not in the type surface. Typing only
 * the two constructors used keeps that dependency-free arrangement while still compiling.
 */
const runtime = (
  globalThis as unknown as {
    WebAssembly: {
      Module: new (bytes: Uint8Array) => unknown;
      Instance: new (module: unknown) => { exports: Record<string, unknown> };
    };
  }
).WebAssembly;

const run = (bytes: Uint8Array): bigint => {
  const instance = new runtime.Instance(new runtime.Module(translateBytes(bytes)));
  const entry = instance.exports.main as () => bigint;
  return entry();
};

const refuse = (bytes: Uint8Array): Error => {
  try {
    translateBytes(bytes);
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected the translator to refuse this program");
};

describe("a counted loop", () => {
  it("runs its body five times and returns the count, which is a decrease-and-test loop", () => {
    // mov $5,%eax ; xor %ecx,%ecx ; add $1,%ecx ; sub $1,%eax ; cmp $0,%eax ; jne head ; mov %ecx,%eax ; ret
    expect(run(hex("b8 05 00 00 00 31 c9 83 c1 01 83 e8 01 83 f8 00 75 f5 89 c8 c3"))).toBe(5n);
  });

  it("accumulates 1 + 2 + 3 + 4, which is a test-at-the-top loop", () => {
    // xor %eax,%eax ; xor %ecx,%ecx ; head: add $1,%ecx ; add %ecx,%eax ; cmp $4,%ecx ; jl head ; ret
    expect(run(hex("31 c0 31 c9 83 c1 01 01 c8 83 f9 04 7c f6 c3"))).toBe(10n);
  });

  it("leaves from the middle of the loop, which needs a block around it", () => {
    /*
     * xor %eax,%eax ; xor %ecx,%ecx ; head: add $1,%ecx ; add %ecx,%eax ; cmp $9,%eax ; jge done ;
     * cmp $10,%ecx ; jl head ; done: ret
     * the sum reaches 10 at the fourth pass, and the exit is a branch to the instruction after the loop
     */
    expect(run(hex("31 c0 31 c9 83 c1 01 01 c8 83 f8 09 7d 05 83 f9 0a 7c f1 c3"))).toBe(10n);
  });

  it("loops on an unconditional backward jmp, with the exit at the top", () => {
    // head: add $1,%ecx ; cmp $5,%ecx ; je done ; jmp head ; done: mov %ecx,%eax ; ret
    expect(run(hex("83 c1 01 83 f9 05 74 02 eb f6 89 c8 c3"))).toBe(5n);
  });

  it("nests one loop inside another, which is a loop whose body opens another", () => {
    // the inner loop adds two per pass and the outer runs three times, so arithmetically 6
    expect(run(hex("31 c0 31 c9 31 d2 83 c2 01 83 c0 01 83 fa 02 7c f5 83 c1 01 83 f9 03 7c eb c3"))).toBe(6n);
  });
});

describe("what a loop refuses", () => {
  it("expresses a jump into the middle of a loop, by naming blocks rather than nesting them", () => {
    /*
     * `cmp $1,%eax; jne +0; jmp -2; ret`: a `jmp` back into the middle of the bytes above it, which no nesting of blocks can
     * express, and which used to be refused. It is translated as a block number now, and the jump this program takes is the one
     * back to itself, so it is only translated here and never run: an unbounded loop is a program that does not stop, and the
     * point of the assertion is that the translator no longer has an opinion about that.
     */
    const program = hex("83 f8 01 75 00 eb fe c3");
    expect(translateBytes(program).length).toBeGreaterThan(0);
  });

  it("runs a loop, with the branch taken back to the block that counts", () => {
    // mov $0,%eax ; add $1,%eax ; cmp $3,%eax ; jne -8 ; ret, which counts to three and answers 3
    expect(run(hex("b8 00 00 00 00 83 c0 01 83 f8 03 75 f8 c3"))).toBe(3n);
  });

  it("refuses a branch whose target is not an instruction boundary at all", () => {
    // cmp $1,%eax ; jne -3 : the displacement lands in the middle of the cmp, so no structure can express it
    expect(refuse(hex("83 f8 01 75 fd c3")).message).toContain("not an instruction boundary");
  });
});

describe("the 32-bit narrowing bug a loop found", () => {
  it("narrows a 32-bit register source as well as the destination", () => {
    /*
     * xor %eax,%eax ; xor %ecx,%ecx ; add $1,%ecx ; add %ecx,%eax ; cmp $3,%ecx ; jl -10 ; ret
     * `add %ecx,%eax` is the plain form with a register source, and before this test existed only the immediate form was
     * covered, so the source arrived as an i64 next to an i32 destination and the module failed to validate
     */
    expect(run(hex("31 c0 31 c9 83 c1 01 01 c8 83 f9 03 7c f6 c3"))).toBe(6n);
  });

  it("keeps a 32-bit add wrapping at 32 bits when the source is a register", () => {
    // mov $-1,%ecx ; add %ecx,%eax  ->  eax is 0xffffffff, not -1 widened into the upper half
    expect(run(hex("b9 ff ff ff ff 01 c8 c3"))).toBe(0xffffffffn);
  });
});

describe("an entry point", () => {
  it("ends in hlt, which traps rather than returning a value", () => {
    // `mov $7,%eax ; hlt`: %eax is written and then the machine traps, which is what `hlt` does in user mode
    expect(() => run(hex("b8 07 00 00 00 f4"))).toThrow();
  });

  it("asks the machine what it can do and is told nothing optional", () => {
    /*
     * `cpuid` answers into four registers from the leaf in %eax, and this machine's answer is zero for every leaf: no vendor is
     * reported and nothing beyond the base leaf is offered, which is a real x86-64 without the optional features. A leaf of 1 is
     * requested here and %edx is copied out to show the answer is zero rather than whatever the request put in the register.
     */
    expect(run(hex("b8 01 00 00 00 0f a2 48 89 d0 c3"))).toBe(0n);
  });
});
