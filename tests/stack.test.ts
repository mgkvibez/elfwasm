/**
 * The stack: a pointer the translator initialises, pushes and pops that move it, and a floor it may not cross.
 *
 * The stack lives in the imported memory above the image, so the caller allocates it and the module addresses it. That is a
 * deliberate consequence of the module importing its memory rather than owning one: a stack the module allocated for itself
 * would be a stack nothing else could see or size.
 */

import { describe, expect, it } from "vitest";

import { addressSpaceOf } from "../src/image.js";
import type { MemoryImage } from "../src/image.js";
import { translateBytes } from "../src/translate.js";

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

/** One 64 KiB page of image at 0x400000, so a stack of n pages sits at virtual 0x410000 and above. */
function image(): MemoryImage {
  return {
    baseAddress: 0x400000n,
    wasmBytes: 0x10000,
    pages: 1,
    entryAddress: 0,
    segments: [
      {
        virtualAddress: 0x400000n,
        fileOffset: 0n,
        wasmAddress: 0,
        fileSize: 0x200n,
        memorySize: 0x200n,
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

const STACK_PAGES = 2;
const TOTAL_PAGES = 1 + STACK_PAGES;
/** The top of the imported memory in WASM offsets, which is where `%rsp` starts. */
const TOP = TOTAL_PAGES * 0x10000;

function run(bytes: Uint8Array, stackPages = STACK_PAGES): Run {
  const memory = new runtime.Memory({ initial: TOTAL_PAGES });
  const buffer = new Uint8Array(memory.buffer);
  const module = translateBytes(bytes, "main", {
    addresses: addressSpaceOf(image()),
    stackPages,
    memoryPages: TOTAL_PAGES,
  });
  const instance = new runtime.Instance(new runtime.Module(module), { env: { memory } });
  return { result: instance.exports.main!(), buffer };
}

function attempt(bytes: Uint8Array): bigint | Error {
  try {
    return run(bytes).result;
  } catch (error) {
    return error as Error;
  }
}

describe("push and pop", () => {
  it("writes below the top of memory and reads the value back", () => {
    // mov $5, %rax ; push %rax ; pop %rbx ; mov %rbx, %rax ; ret
    const { result, buffer } = run(hex("48 c7 c0 05 00 00 00 50 5b 48 89 d8 c3"));
    expect(result).toBe(5n);
    // the value is still where it was left, one word below the top, which is the layout and not just the arithmetic
    expect(Array.from(buffer.subarray(TOP - 8, TOP))).toEqual([5, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("pushes an immediate and pops it into a 32-bit register", () => {
    // push $0x7 ; pop %eax ; ret
    const { result } = run(hex("6a 07 58 c3"));
    expect(result).toBe(7n);
  });

  it("reads through the pointer that push moved", () => {
    // mov $0x11, %rax ; push %rax ; mov (%rsp), %rax ; ret
    const { result } = run(hex("48 c7 c0 11 00 00 00 50 48 8b 04 24 c3"));
    expect(result).toBe(0x11n);
  });

  it("runs the frame-pointer idiom a compiled function opens with", () => {
    // push %rbp ; mov %rsp,%rbp ; sub $0x10,%rsp ; movq $0x22,-8(%rbp) ; mov -8(%rbp),%rax ; leave ; ret
    const { result } = run(
      hex("55 48 89 e5 48 83 ec 10 48 c7 45 f8 22 00 00 00 48 8b 45 f8 c9 c3"),
    );
    expect(result).toBe(0x22n);
  });

  it("leaves %rsp where the frame pointer said it was", () => {
    // push %rbp ; mov %rsp,%rbp ; sub $0x40,%rsp ; push %rax ; leave ; mov %rsp,%rax ; ret
    // after leave, rsp must be the top of memory again: the pushed word is gone from the accounting even though the byte
    // is still there, which is what makes a frame re-entrant
    const { result } = run(hex("55 48 89 e5 48 83 ec 40 50 c9 48 89 e0 c3"));
    expect(result).toBe(BigInt(0x400000 + TOP));
  });
});

describe("the stack floor", () => {
  it("traps instead of growing down into the image", () => {
    // sub $0x100000, %rsp ; push %rax ; ret: the push would land well inside the image
    expect(attempt(hex("48 81 ec 00 00 10 00 50 c3"))).toBeInstanceOf(Error);
  });

  it("allows a push at the lowest legal word", () => {
    // sub $0x1fff8, %rsp puts rsp at the first word above the image, which is exactly where the floor begins
    const { result, buffer } = run(hex("48 81 ec f8 ff 01 00 48 c7 c0 07 00 00 00 50 58 c3"));
    expect(result).toBe(7n);
    expect(buffer[0x10000]).toBe(7);
  });
});

describe("opting in", () => {
  it("refuses a function that uses the stack when no pages were set aside", () => {
    // push %rax ; pop %rax ; ret, with no stackPages: refusing is right, because the import minimum would be too small
    try {
      translateBytes(hex("50 58 c3"), "main", { addresses: addressSpaceOf(image()) });
    } catch (error) {
      expect((error as Error).message).toContain("needs stackPages");
      return;
    }
    throw new Error("expected a refusal");
  });

  it("counts the stack pages in the memory the module imports", () => {
    const image_ = image();
    const module = translateBytes(hex("50 58 c3"), "main", {
      addresses: addressSpaceOf(image_),
      stackPages: 2,
      memoryPages: 3,
    });
    const compiled = new runtime.Module(module);
    // a caller that supplies only the image's page is refused by the engine, which is the loud version of a quiet bug
    expect(() => new runtime.Instance(compiled, { env: { memory: new runtime.Memory({ initial: 1 }) } })).toThrow();
    const memory = new runtime.Memory({ initial: 3 });
    new runtime.Instance(compiled, { env: { memory } });
    expect(new Uint8Array(memory.buffer)[TOP - 8]).toBe(0);
  });

  it("defaults the page count to the image plus the stack", () => {
    // no memoryPages: the module must still ask for three pages, or instantiation against the caller's memory fails
    const module = translateBytes(hex("50 58 c3"), "main", {
      addresses: addressSpaceOf(image()),
      stackPages: 2,
    });
    const memory = new runtime.Memory({ initial: 3 });
    new runtime.Instance(new runtime.Module(module), { env: { memory } });
  });
});

describe("call", () => {
  it("is refused here, because a single function has no program to call into", () => {
    // call +0
    try {
      translateBytes(hex("e8 00 00 00 00 c3"), "main", {
        addresses: addressSpaceOf(image()),
        stackPages: 1,
        memoryPages: 2,
      });
    } catch (error) {
      expect((error as Error).message).toContain("needs the code region");
      return;
    }
    throw new Error("expected a refusal");
  });
});
