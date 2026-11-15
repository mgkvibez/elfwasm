/**
 * The two shuffles, and why their expectations are written down rather than measured.
 *
 * Every other check in this repository that can compare against the processor does. These two cannot, and the reason is recorded in
 * `tools/native/native.md`: the host these tests run on is gVisor, and its answers for `shufps` are not the architecture's. Two of its
 * own answers contradict each other, since the same selector read through the same instruction returned the destination's lane 3 for
 * one immediate and the source's lane 2 for another, so it cannot arbitrate between one reading of the manual and another. The
 * expectations below are therefore `_mm_shuffle_ps` and `_mm_shuffle_pd` written out, which is the definition a compiler implements.
 */

import { describe, expect, it } from "vitest";

import { decode } from "../src/decode.js";
import { translate } from "../src/translate.js";

const runtime = (
  globalThis as unknown as {
    WebAssembly: {
      Module: new (bytes: Uint8Array) => unknown;
      Instance: new (module: unknown) => { exports: Record<string, () => bigint> };
    };
  }
).WebAssembly;

const hex = (text: string): Uint8Array =>
  Uint8Array.from(text.trim().split(/\s+/).map((pair) => Number.parseInt(pair, 16)));

/** The eight bytes of a value, little end first, which is how `movabs` carries an immediate. */
function le64(value: bigint): string {
  return Array.from({ length: 8 }, (_, index) => Number((value >> BigInt(8 * index)) & 0xffn).toString(16).padStart(2, "0")).join(" ");
}

/**
 * Build a program, translate it, and read `%rax` back.
 *
 * A vector here is two quadwords in two halves, so a test builds one the way a compiler does: `movq` into the low half, then
 * `punpcklqdq` to bring a second quadword into the high one. A vector literal would be sixteen bytes of memory, and the subset has no
 * instruction that loads that.
 */
function run(parts: string[]): bigint {
  const instance = new runtime.Instance(new runtime.Module(translate(decode(hex(parts.join(" "))))));
  return instance.exports.main!();
}

// the vector register is the ModRM reg field of `movq %rax,%xmm`, so it is shifted into place rather than added
const load = (value: bigint, register: number): string => `48 b8 ${le64(value)}` + ` 66 48 0f 6e ${(0xc0 + (register << 3)).toString(16)}`;
const SHUFPD = (imm: number): string => `66 0f c6 c1 ${imm.toString(16).padStart(2, "0")}`;
const SHUFPS = (imm: number): string => `0f c6 c1 ${imm.toString(16).padStart(2, "0")}`;
const OUT_RAX = "66 48 0f 7e c0 c3";

/** The destination `[1, 2, 3, 4]`, built from two quadwords: the low lane pair and the high one. */
const DESTINATION = [load(0x0000000200000001n, 0), load(0x0000000400000003n, 6), "66 0f 6c c6"];

/** The source `[5, 6, 7, 8]`, in `%xmm1`. */
const SOURCE = [load(0x0000000600000005n, 1), load(0x0000000800000007n, 6), "66 0f 6c ce"];

/** The low quadword of the destination, which is what a store or a `movq` can see. */
const lowOf = (body: string[]): bigint => run([...DESTINATION, ...SOURCE, ...body, OUT_RAX]);

/** The high quadword, brought down into the low half by `punpckhqdq` against itself before it is read. */
const highOf = (body: string[]): bigint => run([...DESTINATION, ...SOURCE, ...body, "66 0f 6d c0", OUT_RAX]);

describe("shufpd", () => {
  it("takes each half from one operand, chosen by one bit", () => {
    // bit 0 set: the low half is the destination's high quadword, and this is the case that a reader expecting a no-op gets wrong
    expect(lowOf([SHUFPD(1)])).toBe(0x0000000400000003n);
    // bit 0 clear keeps the destination's low half
    expect(lowOf([SHUFPD(2)])).toBe(0x0000000200000001n);
    // bit 1 set: the high half is the source's high quadword
    expect(highOf([SHUFPD(2)])).toBe(0x0000000800000007n);
    // bit 1 clear: the high half is the source's low quadword, so `$0` is not an identity
    expect(highOf([SHUFPD(0)])).toBe(0x0000000600000005n);
    // both bits: the destination's high half and the source's high half
    expect(lowOf([SHUFPD(3)])).toBe(0x0000000400000003n);
    expect(highOf([SHUFPD(3)])).toBe(0x0000000800000007n);
  });

  it("survives being its own source", () => {
    // `shufpd $1,%xmm0,%xmm0` moves the high half down, which a write-then-read ordering would answer with the wrong quadword
    const moved = run([...DESTINATION, "66 0f c6 c0 01", OUT_RAX]);
    expect(moved).toBe(0x0000000400000003n);
  });
});

describe("shufps", () => {
  it("takes lanes 0 and 1 from the destination and 2 and 3 from the source", () => {
    // 0x1b is 00 01 10 11, so the four selectors are 3, 2, 1, 0: source lane 1, source lane 0, destination lane 1, destination lane 0
    expect(lowOf([SHUFPS(0x1b)])).toBe(0x0000000500000006n);
    expect(highOf([SHUFPS(0x1b)])).toBe(0x0000000100000002n);
  });

  it("keeps both lanes of each selection when the selectors repeat", () => {
    // 0xe4 is 11 10 01 00: both destination lanes, then both source lanes, which is the identity shuffle across two registers
    expect(lowOf([SHUFPS(0xe4)])).toBe(0x0000000200000001n);
    expect(highOf([SHUFPS(0xe4)])).toBe(0x0000000600000005n);
    // 0x00 names the destination's low lane four times, which is a broadcast of one dword
    expect(lowOf([SHUFPS(0x00)])).toBe(0x0000000100000001n);
  });
});

describe("the shuffles refused", () => {
  it("names the memory form rather than translating half of it", () => {
    expect(() => decode(hex("66 0f c6 01 01"))).toThrow(/shufpd from memory/);
    expect(() => decode(hex("0f c6 01 1b"))).toThrow(/shufps from memory/);
  });
});
