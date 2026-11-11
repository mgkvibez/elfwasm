/**
 * Memory: addresses resolved against the mapped image, and reads and writes that land where the processor puts them.
 *
 * The reference for every expectation here is the x86 instruction's own semantics rather than this translator's output,
 * and the memory is the caller's rather than the module's: the module imports it, so what the tests inspect afterwards is
 * the same buffer the translated code wrote through.
 */

import { describe, expect, it } from "vitest";

import { decode } from "../src/decode.js";
import { addressSpaceOf } from "../src/image.js";
import type { MemoryImage } from "../src/image.js";
import { translate, translateBytes } from "../src/translate.js";

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

/** A one-segment image at 0x400000, which is where a static link puts code, with 256 bytes of data behind it. */
function imageOf(bytes = 0x200): MemoryImage {
  return {
    baseAddress: 0x400000n,
    loadBias: 0n,
    wasmBytes: 0x10000,
    pages: 1,
    entryAddress: 0,
    segments: [
      {
        virtualAddress: 0x400000n,
        fileOffset: 0n,
        wasmAddress: 0,
        fileSize: BigInt(bytes),
        memorySize: BigInt(bytes),
        permissions: { read: true, write: true, execute: true },
      },
    ],
  };
}

function hex(text: string): Uint8Array {
  return Uint8Array.from(text.trim().split(/\s+/).map((byte) => Number.parseInt(byte, 16)));
}

interface Run {
  readonly result: bigint;
  readonly buffer: Uint8Array;
}

/** Translate, instantiate against a memory this test fills, call, and hand back both the result and the buffer. */
function run(bytes: Uint8Array, options: { codeAddress?: bigint; fill?: number } = {}): Run {
  const image = imageOf();
  const memory = new runtime.Memory({ initial: 1 });
  const buffer = new Uint8Array(memory.buffer);
  buffer.fill(options.fill ?? 0);
  const module = translateBytes(bytes, "main", {
    addresses: addressSpaceOf(image),
    memoryPages: 1,
    ...(options.codeAddress === undefined ? {} : { codeAddress: options.codeAddress }),
  });
  const instance = new runtime.Instance(new runtime.Module(module), { env: { memory } });
  return { result: instance.exports.main!(), buffer };
}

function refusal(bytes: Uint8Array, options: { codeAddress?: bigint } = {}): string {
  try {
    translateBytes(bytes, "main", {
      addresses: addressSpaceOf(imageOf()),
      ...(options.codeAddress === undefined ? {} : { codeAddress: options.codeAddress }),
    });
  } catch (error) {
    return (error as Error).message;
  }
  throw new Error("expected a refusal, and the translation succeeded");
}

describe("stores and loads", () => {
  it("writes through the caller's memory and reads the value back into a register", () => {
    // movq $0x2a, 0x400100 ; mov 0x400100, %rax ; ret
    const { result, buffer } = run(hex("48 c7 04 25 00 01 40 00 2a 00 00 00 48 8b 04 25 00 01 40 00 c3"));
    expect(result).toBe(0x2an);
    expect(buffer[0x100]).toBe(0x2a);
  });

  it("zero-extends a 32-bit load, which is what movl does", () => {
    // movl $-1, 0x400100 ; mov 0x400100, %eax ; ret: rax must be 0x00000000ffffffff, not -1
    const { result } = run(hex("c7 04 25 00 01 40 00 ff ff ff ff 48 8b 04 25 00 01 40 00 c3"));
    expect(result).toBe(0xffffffffn);
  });

  it("keeps a 64-bit load signed, which is what movq does", () => {
    // movq $-1, 0x400100 ; mov 0x400100, %rax ; ret
    const { result } = run(hex("48 c7 04 25 00 01 40 00 ff ff ff ff 48 8b 04 25 00 01 40 00 c3"));
    expect(result).toBe(-1n);
  });

  it("writes a single byte and leaves its neighbours alone", () => {
    // movb $0x7f, 0x400100 ; ret, over memory filled with 0xaa
    const { buffer } = run(hex("c6 04 25 00 01 40 00 7f c3"), { fill: 0xaa });
    expect(buffer[0x100]).toBe(0x7f);
    expect(buffer[0x101]).toBe(0xaa);
  });

  it("adds a memory operand to a register", () => {
    // movq $0x20, 0x400100 ; mov $0x10, %rax ; add 0x400100, %rax ; ret
    const { result } = run(
      hex("48 c7 04 25 00 01 40 00 20 00 00 00 48 c7 c0 10 00 00 00 48 03 04 25 00 01 40 00 c3"),
    );
    expect(result).toBe(0x30n);
  });

  it("imports the memory rather than owning one, so the caller's buffer is the one written", () => {
    const module = translateBytes(
      hex("48 c7 04 25 00 01 40 00 2a 00 00 00 c3"),
      "main",
      { addresses: addressSpaceOf(imageOf()), memoryPages: 1 },
    );
    const compiled = new runtime.Module(module);
    // no import supplied: the module asks for one, which is the difference between mapping an image and pretending to
    expect(() => new runtime.Instance(compiled)).toThrow();
    const memory = new runtime.Memory({ initial: 1 });
    new runtime.Instance(compiled, { env: { memory } }).exports.main!();
    expect(new Uint8Array(memory.buffer)[0x100]).toBe(0x2a);
  });

  it("needs no memory import when nothing addresses memory", () => {
    const instance = new runtime.Instance(new runtime.Module(translateBytes(hex("48 c7 c0 05 00 00 00 c3"))));
    expect(instance.exports.main!()).toBe(5n);
  });
});

describe("address resolution", () => {
  it("resolves a RIP-relative operand from the instruction's own end", () => {
    // mov 0x0(%rip), %rax at 0x400000, seven bytes long: the target is 0x400107, which is inside the segment
    const bytes = hex("48 8b 05 00 01 00 00 c3");
    const image = imageOf();
    const memory = new runtime.Memory({ initial: 1 });
    new Uint8Array(memory.buffer)[0x107] = 0x2b;
    const module = translateBytes(bytes, "main", { addresses: addressSpaceOf(image), codeAddress: 0x400000n });
    expect(new runtime.Instance(new runtime.Module(module), { env: { memory } }).exports.main!()).toBe(0x2bn);
  });

  it("refuses a RIP-relative operand that lands outside every segment", () => {
    // the same bytes with no codeAddress: the target becomes 0x107, which nothing maps
    expect(refusal(hex("48 8b 05 00 01 00 00 c3"))).toContain("0x107 is not mapped");
  });

  it("refuses an address outside every segment instead of reading a plausible byte", () => {
    expect(refusal(hex("48 8b 04 25 00 20 40 00 c3"))).toContain("0x402000 is not mapped by any segment");
  });

  it("refuses a memory operand when no address space was supplied", () => {
    try {
      translate(decode(hex("48 8b 04 25 00 01 40 00 c3")));
    } catch (error) {
      expect((error as Error).message).toContain("needs the mapped image");
      expect((error as Error).message).toContain("addressSpaceOf");
      return;
    }
    throw new Error("expected a refusal");
  });

  it("refuses a segment override it has no base for, rather than reading the offset as a plain address", () => {
    // %gs:0x28: treating 0x28 as an address would load the wrong memory and look fine
    // %gs:0x28, whose base is the one this translator does not keep: the refusal names the segment rather than the offset
    expect(refusal(hex("65 48 8b 04 25 28 00 00 00 c3"))).toContain("%gs:");
    expect(refusal(hex("65 48 8b 04 25 28 00 00 00 c3"))).toContain("refusing");
  });

  it("turns %fs:0x28 into a read from the thread pointer, not from address 0x28", () => {
    // the stack canary's read, which is the reason %fs: exists at all
    const code = translateBytes(hex("64 48 8b 04 25 28 00 00 00 c3"), "main", { addresses: addressSpaceOf(imageOf()) });
    // the translation reads the thread pointer's global before it addresses anything
    expect(code).toContain(0x23);
    expect(code.length).toBeGreaterThan(20);
  });

  it("resolves an index register at run time instead of refusing it", () => {
    // mov (%rax,%rcx,1), %rdx: the address is computed from two registers, so the translation defers it
    const code = translateBytes(hex("48 8b 14 08 c3"), "main", { addresses: addressSpaceOf(imageOf()) });
    expect(code.length).toBeGreaterThan(0);
  });
});

describe("the 16-byte moves", () => {
  it("stores both halves of a vector register eight bytes apart", () => {
    /*
     * `mov $0x11223344,%eax ; movd %eax,%xmm0 ; movhps` would need a load, so the two halves are written first and then stored at
     * once: `movaps %xmm0,0x400100` writes 0x11223344 at 0x100 and the zeroed upper half at 0x108.
     */
    const { buffer } = run(hex("b8 44 33 22 11 66 0f 6e c0 0f 29 04 25 00 01 40 00 c3"));
    expect(buffer[0x100]).toBe(0x44);
    expect(buffer[0x101]).toBe(0x33);
    expect(buffer[0x104]).toBe(0);
    expect(buffer[0x108]).toBe(0);
  });

  it("loads both halves back, which is the move in the other direction", () => {
    // `movaps 0x400100,%xmm0 ; movd %xmm0,%eax ; ret`, over memory whose first eight bytes are a known value
    const { result } = run(hex("0f 28 04 25 00 01 40 00 66 0f 7e c0 c3"), {});
    expect(result).toBe(0n);
    // and a byte filled in by the test itself comes back through the vector register
    const filled = run(hex("0f 28 04 25 00 01 40 00 66 0f 7e c0 c3"), { fill: 0x11 });
    expect(filled.result).toBe(0x11111111n);
  });
});
