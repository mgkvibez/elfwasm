/**
 * Addressing through a register, where the address only exists once the code runs.
 *
 * This is the increment that makes real compiler output reachable: a pointer is a register holding a virtual address, and
 * the translation has to subtract the image's base and then decide whether the result is inside the memory at all. WASM has
 * one flat buffer and no faulting pages, so an address outside the image would read whatever sits at that offset and look
 * entirely legitimate; every check here that expects a trap exists to keep that from being an acceptable outcome.
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

/** One segment at 0x400000, one 64 KiB page, and 0x200 bytes of it file-backed. */
function image(): MemoryImage {
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

/**
 * Run against a memory this test prefills, and hand back both the result and the buffer afterwards.
 *
 * `codeAddress` is only needed by a test whose operand is RIP-relative: without it the bytes are assumed to start at address
 * zero, which is fine for everything that names its addresses outright.
 */
function run(
  bytes: Uint8Array,
  fill?: (buffer: Uint8Array) => void,
  codeAddress?: bigint,
): { result: bigint; buffer: Uint8Array } {
  const mapped = image();
  const memory = new runtime.Memory({ initial: 1 });
  const buffer = new Uint8Array(memory.buffer);
  buffer.fill(0);
  fill?.(buffer);
  const module = translateBytes(bytes, "main", {
    addresses: addressSpaceOf(mapped),
    memoryPages: 1,
    ...(codeAddress === undefined ? {} : { codeAddress }),
  });
  const instance = new runtime.Instance(new runtime.Module(module), { env: { memory } });
  return { result: instance.exports.main!(), buffer };
}

/** The same translation, called, with whatever it throws returned instead. */
function attempt(bytes: Uint8Array, codeAddress?: bigint): bigint | Error {
  try {
    return run(bytes, undefined, codeAddress).result;
  } catch (error) {
    return error as Error;
  }
}

describe("addressing through a base register", () => {
  it("loads through a register holding a virtual address", () => {
    // mov $0x400100, %rcx ; mov (%rcx), %rax ; ret
    const { result } = run(hex("48 c7 c1 00 01 40 00 48 8b 01 c3"), (buffer) => {
      buffer[0x100] = 0x2c;
    });
    expect(result).toBe(0x2cn);
  });

  it("stores through a register, at the address the image maps", () => {
    // mov $0x400100, %rcx ; movq $9, (%rcx) ; ret
    const { buffer } = run(hex("48 c7 c1 00 01 40 00 48 c7 01 09 00 00 00 c3"));
    expect(Array.from(buffer.subarray(0x100, 0x108))).toEqual([9, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("adds a displacement, positive and negative", () => {
    // mov $0x400100, %rcx ; mov $0x11, 8(%rcx) is not needed: prefill instead, then read at +8 and -8
    const forward = run(hex("48 c7 c1 00 01 40 00 48 8b 41 08 c3"), (buffer) => {
      buffer[0x108] = 0x31;
    });
    expect(forward.result).toBe(0x31n);
    // mov $0x400108, %rcx ; mov -8(%rcx), %rax ; ret: the same byte, addressed backwards
    const backward = run(hex("48 c7 c1 08 01 40 00 48 8b 41 f8 c3"), (buffer) => {
      buffer[0x100] = 0x32;
    });
    expect(backward.result).toBe(0x32n);
  });

  it("scales an index register, which is what an array access is", () => {
    // mov $0x400100, %rcx ; mov $3, %rax ; mov (%rcx,%rax,4), %rdx ; mov %rdx, %rax ; ret
    const { result } = run(
      hex("48 c7 c1 00 01 40 00 48 c7 c0 03 00 00 00 48 8b 14 81 48 89 d0 c3"),
      (buffer) => {
        buffer[0x10c] = 0x51;
      },
    );
    expect(result).toBe(0x51n);
  });

  it("keeps the compare operand safe while a second operand is loaded through a register", () => {
    // the destination of a compare lives in a local of its own, so an address computation must not use that local:
    // mov $0x400100,%rcx ; movq $5,(%rcx) ; mov $5,%rdx ; mov $0,%rax ; cmp (%rcx),%rdx ; jne +7 ; mov $0x2a,%rax ; ret
    const { result } = run(
      hex("48 c7 c1 00 01 40 00 48 c7 01 05 00 00 00 48 c7 c2 05 00 00 00 48 c7 c0 00 00 00 00 48 3b 11 75 07 48 c7 c0 2a 00 00 00 c3"),
    );
    expect(result).toBe(0x2an);
  });
});

describe("an address outside the image", () => {
  it("traps rather than reading whatever is at that offset", () => {
    // mov $0x900000, %rcx ; mov (%rcx), %rax ; ret
    expect(attempt(hex("48 c7 c1 00 00 90 00 48 8b 01 c3"))).toBeInstanceOf(Error);
  });

  it("traps on a null pointer, which is a negative offset once the base is subtracted", () => {
    // xor %rcx, %rcx ; mov (%rcx), %rax ; ret
    expect(attempt(hex("48 31 c9 48 8b 01 c3"))).toBeInstanceOf(Error);
  });

  it("traps when the last bytes of the image are not wide enough for the access", () => {
    // a 64-bit store at the very end of the 64 KiB page: the address is inside, the eight bytes are not
    // mov $0x40fffe, %rcx ; movq $1, (%rcx) ; ret
    expect(attempt(hex("48 c7 c1 fe ff 40 00 48 c7 01 01 00 00 00 c3"))).toBeInstanceOf(Error);
  });

  it("allows the last bytes that do fit", () => {
    // the same store eight bytes earlier succeeds, so the check is a bound and not a blanket refusal
    const { buffer } = run(hex("48 c7 c1 f8 ff 40 00 48 c7 01 01 00 00 00 c3"));
    expect(buffer[0xfff8]).toBe(1);
  });
});

describe("refusals that remain", () => {
  it("still needs the image even for a register-relative address", () => {
    try {
      translateBytes(hex("48 8b 01 c3"));
    } catch (error) {
      expect((error as Error).message).toContain("needs the mapped image");
      return;
    }
    throw new Error("expected a refusal");
  });

  it("still refuses a segment override whose base is not kept", () => {
    // %gs:(%rcx): the thread pointer is kept for %fs only, and reading the offset as a plain address would be a wrong answer
    try {
      translateBytes(hex("65 48 8b 01 c3"), "main", { addresses: addressSpaceOf(image()) });
    } catch (error) {
      expect((error as Error).message).toContain("a %gs: operand needs the %gs base");
      return;
    }
    throw new Error("expected a refusal");
  });
});

describe("lea, which names an address instead of reading one", () => {
  it("puts a virtual address in a register, which a later load rebases", () => {
    /*
     * lea 0x2f8(%rip), %rax ; mov (%rax), %rax ; ret
     *
     * The displacement is relative to the byte after the `lea`, which is at virtual 0x400007, so the address it names is
     * 0x4002ff... and the load that follows has to subtract the image base to reach offset 0x2ff. Both halves are the point:
     * a `lea` that stored an offset would load from the wrong byte, and one that stored a rebased address would trap.
     */
    const { result } = run(
      hex("48 8d 05 f8 02 00 00 48 8b 00 c3"),
      (buffer) => {
        buffer[0x2ff] = 0x5a;
      },
      0x400000n,
    );
    expect(result).toBe(0x5an);
  });

  it("adds a base register and a scaled index without reading memory", () => {
    // lea (%rcx,%rax,4), %rax with %rcx = 0x400100 and %rax = 2, so the result is 0x400108, which holds 7
    const { result } = run(
      hex("48 c7 c1 00 01 40 00 48 c7 c0 02 00 00 00 48 8d 04 81 48 8b 00 c3"),
      (buffer) => {
        buffer[0x108] = 0x07;
      },
    );
    expect(result).toBe(0x07n);
  });

  it("allows an address one past the end of memory, which is how a compiler finds the end of an array", () => {
    // lea 0xfff9(%rip), %rax ; ret: 0x400007 + 0xfff9 is 0x410000, the first byte past the page, and that is a legal value
    expect(attempt(hex("48 8d 05 f9 ff 00 00 c3"), 0x400000n)).toBe(0x410000n);
  });

  it("still traps when that address is then read", () => {
    // the same lea, followed by a load: the value is legal, the access is not
    const outcome = attempt(hex("48 8d 05 f9 ff 00 00 48 8b 00 c3"), 0x400000n);
    expect(outcome).toBeInstanceOf(Error);
  });
});
