/**
 * Rotates: a rotate is a rotate of its own width, its count is reduced to that width, and it does not touch the flags a comparison
 * left.
 *
 * Each expectation is the architecture's own answer rather than this translator's: the swap-halves idiom moves the top dword into the
 * bottom one, a count equal to the width is no rotation, and a branch after a rotate still answers the comparison before it.
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
  Uint8Array.from(text.trim().split(/\s+/).map((part) => Number.parseInt(part, 16)));

/** The value a body returns in `%rax`, translated from the bytes and run, as an unsigned sixty-four-bit number. */
const run = (bytes: Uint8Array): bigint =>
  BigInt.asUintN(64, new runtime.Instance(new runtime.Module(translate(decode(bytes)))).exports.main!());

describe("rotates", () => {
  it("moves each bit by the count, which for the swap-halves idiom is one dword into the other", () => {
    // movabs $0x0123456789abcdef,%rax ; ror $32,%rax ; ret
    expect(run(hex("48 b8 ef cd ab 89 67 45 23 01 48 c1 c8 20 c3"))).toBe(0x89abcdef01234567n);
    // and the same value rotated one place, where the lowest bit goes to the top
    expect(run(hex("48 b8 ef cd ab 89 67 45 23 01 48 d1 c8 c3"))).toBe(0x8091a2b3c4d5e6f7n);
  });

  it("rotates inside its own width, so a byte rotate is a byte's rotation and a count of eight changes nothing", () => {
    // mov $0x11,%al ; ror $1,%al ; ret
    expect(run(hex("b0 11 d0 c8 c3"))).toBe(0x88n);
    // mov $0x11,%al ; ror $8,%al ; ret, where eight is the byte's own width
    expect(run(hex("b0 11 c0 c8 08 c3"))).toBe(0x11n);
  });

  it("masks the count the way the processor does before reducing it to the width", () => {
    // mov $0x12345678,%eax ; mov $33,%cl ; ror %cl,%eax ; ret: thirty-three is one below the five-bit mask
    expect(run(hex("b8 78 56 34 12 b1 21 d3 c8 c3"))).toBe(0x91a2b3cn);
  });

  it("leaves the flags a comparison left alone, so a branch after it still answers that comparison", () => {
    // mov $5,%eax ; cmp $5,%eax ; ror $3,%rax ; jne +5 ; mov $7,%eax ; ret ; mov $9,%eax ; ret
    expect(run(hex("b8 05 00 00 00 83 f8 05 48 c1 c8 03 75 05 b8 07 00 00 00 c3 b8 09 00 00 00 c3"))).toBe(7n);
  });

  it("refuses a carry condition after a rotate, which describes the bits that went round", () => {
    // mov $5,%eax ; cmp $5,%eax ; ror $3,%rax ; jb +5, where the carry is the last bit out rather than the comparison's borrow
    expect(() => translate(decode(hex("b8 05 00 00 00 83 f8 05 48 c1 c8 03 72 05 b8 07 00 00 00 c3")))).toThrow(
      /after a rotate reads a flag the rotate does not define/,
    );
  });
});
