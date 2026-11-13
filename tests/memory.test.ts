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
function run(
  bytes: Uint8Array,
  options: {
    codeAddress?: bigint;
    fill?: number;
    code?: { readonly bytes: Uint8Array; readonly virtualAddress: bigint };
    indirectTargets?: readonly bigint[];
  } = {},
): Run {
  const image = imageOf();
  const memory = new runtime.Memory({ initial: 1 });
  const buffer = new Uint8Array(memory.buffer);
  buffer.fill(options.fill ?? 0);
  const module = translateBytes(bytes, "main", {
    addresses: addressSpaceOf(image),
    memoryPages: 1,
    ...(options.codeAddress === undefined ? {} : { codeAddress: options.codeAddress }),
    ...(options.code === undefined ? {} : { code: options.code }),
    ...(options.indirectTargets === undefined ? {} : { indirectTargets: options.indirectTargets }),
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

  it("writes two bytes at a 16-bit width and leaves the two after them alone", () => {
    // mov $0xabcd,%edx ; mov %dx,0x400100 ; mov 0x400100,%rax ; ret
    const { result, buffer } = run(
      hex("ba cd ab 00 00 66 89 14 25 00 01 40 00 48 8b 04 25 00 01 40 00 c3"),
      { fill: 0xaa },
    );
    // the qword around the two bytes is the 0xaa this test filled memory with, so the load reads 0xaaaa_aaaa_aaaa_abcd
    expect(BigInt.asUintN(64, result)).toBe(0xaaaaaaaaaaaaabcdn);
    expect([buffer[0x100], buffer[0x101], buffer[0x102]]).toEqual([0xcd, 0xab, 0xaa]);
  });

  it("keeps the upper 48 bits of a register a 16-bit move writes", () => {
    // mov $0x0123456789abcdef,%rax ; mov $0x1234,%ax ; ret: only the low half of the register is the destination
    const { result } = run(hex("48 b8 ef cd ab 89 67 45 23 01 66 b8 34 12 c3"));
    expect(result).toBe(0x0123456789ab1234n);
  });

  it("zero-extends a 16-bit load into the register it lands in", () => {
    // mov $0xabcd,%edx ; mov %dx,0x400100 ; mov $-16,%rax ; mov 0x400100,%ax ; ret
    const { result } = run(
      hex("ba cd ab 00 00 66 89 14 25 00 01 40 00 48 c7 c0 f0 ff ff ff 66 8b 04 25 00 01 40 00 c3"),
    );
    // 0xfffffffffffffff0 with its low half replaced by 0xabcd: a 16-bit load widens and does not sign-extend
    expect(BigInt.asUintN(64, result)).toBe(0xffffffffffffabcdn);
  });

  it("adds a memory operand to a register", () => {
    // movq $0x20, 0x400100 ; mov $0x10, %rax ; add 0x400100, %rax ; ret
    const { result } = run(
      hex("48 c7 04 25 00 01 40 00 20 00 00 00 48 c7 c0 10 00 00 00 48 03 04 25 00 01 40 00 c3"),
    );
    expect(result).toBe(0x30n);
  });

  it("folds a global into a register with the reverse-direction or, which names the register as the destination", () => {
    /*
     * `or 0x..(%rip),%eax` is the direction gcc emits to fold a global into a value without loading it first: opcode `0x0b` puts the
     * register in ModRM.reg, which is the destination, and the memory operand in ModRM.rm. Only `0x09`, the direction whose
     * destination is the memory operand, was decoded.
     *
     * movq $0x0f000000, 0x400100 ; mov $0x00ff00ff, %eax ; or 0x400100,%eax ; ret
     */
    const { result } = run(
      hex("48 c7 04 25 00 01 40 00 00 00 00 0f b8 ff 00 ff 00 0b 04 25 00 01 40 00 c3"),
    );
    expect(BigInt.asUintN(32, result)).toBe(0x0fff00ffn);
  });

  it("does the same for and, whose reverse direction is opcode 0x23", () => {
    // movq $0x0f000000, 0x400100 ; mov $0xffffffff, %eax ; and 0x400100,%eax ; ret
    const { result } = run(
      hex("48 c7 04 25 00 01 40 00 00 00 00 0f b8 ff ff ff ff 23 04 25 00 01 40 00 c3"),
    );
    expect(BigInt.asUintN(32, result)).toBe(0x0f000000n);
  });

  it("tests a bit of a word in memory, which is how a compiled program reads a flags word", () => {
    /*
     * movq $0x10,0x400100 ; mov $4,%ecx ; bt %rcx,0x400100 ; jb done ; mov $0,%eax ; ret ; done: mov $1,%eax ; ret
     *
     * Bit 4 of 0x10 is set, so the carry is set and the branch is taken. A bit test is the one instruction besides a comparison
     * whose result is a flag, so the fixture is here: the value has to be in memory for the memory form to be the one under test.
     */
    const { result } = run(
      hex("48 c7 04 25 00 01 40 00 10 00 00 00 b9 04 00 00 00 48 0f a3 0c 25 00 01 40 00 72 06 b8 00 00 00 00 c3 b8 01 00 00 00 c3"),
    );
    expect(result).toBe(1n);
  });

  it("folds a register into a word in memory, which is the read-modify-write direction", () => {
    // mov [0x400100],0 ; mov $5,%ecx ; or %ecx,[0x400100] ; mov %eax,[0x400100] ; ret
    const { result } = run(hex("c7 04 25 00 01 40 00 00 00 00 00 b9 05 00 00 00 09 0c 25 00 01 40 00 8b 04 25 00 01 40 00 c3"));
    expect(result).toBe(5n);
  });

  it("subtracts a register from a word in memory and keeps the flags the way a cmp would", () => {
    // mov [0x400100],8 ; mov $8,%ecx ; sub %ecx,[0x400100] ; mov %eax,[0x400100] ; ret  ->  0, and the zero flag is set
    const { result } = run(hex("c7 04 25 00 01 40 00 08 00 00 00 b9 08 00 00 00 29 0c 25 00 01 40 00 8b 04 25 00 01 40 00 c3"));
    expect(result).toBe(0n);
  });

  it("folds an immediate into a word in memory, twice over, which is how a flags word is built", () => {
    // mov [0x400100],0 ; or [0x400100],0x30 ; and [0x400100],0x10 ; mov %eax,[0x400100] ; ret
    const { result } = run(hex("c7 04 25 00 01 40 00 00 00 00 00 83 0c 25 00 01 40 00 30 83 24 25 00 01 40 00 10 8b 04 25 00 01 40 00 c3"));
    expect(result).toBe(0x10n);
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

  it("interleaves two low quadwords, which is the destination's low half moving up and the source's coming in", () => {
    /*
     * mov $0xaaaa,%rax ; movd %rax,%xmm0 ; mov $0xbbbb,%rax ; movd %rax,%xmm1 ; punpcklqdq %xmm1,%xmm0 ; movaps %xmm0,0x400100
     *
     * Both vector registers are built from a `movd`, which zeroes the half it does not write, so the interleave is visible: the low
     * half of the result is the source's 0xbbbb and the high half is the destination's old 0xaaaa.
     */
    const { buffer } = run(
      hex("48 c7 c0 aa 0a 00 00 66 0f 6e c0 48 c7 c0 bb 0b 00 00 66 0f 6e c8 66 0f 6c c1 0f 29 04 25 00 01 40 00 c3"),
    );
    expect([...buffer.slice(0x100, 0x110)].map((byte) => byte.toString(16).padStart(2, "0")).join("")).toBe(
      "bb0b000000000000aa0a000000000000",
    );
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

describe("an indirect call", () => {
  /*
   * The target is a value the program reads rather than one the linker wrote down, so which function runs is decided at run time.
   * `call *0x400100` reads a pointer the program stored, and the function it names is one nothing calls directly, which is what
   * `indirectTargets` is for: without it there would be nothing to compare the pointer against.
   */
  const program = hex(
    [
      "48 c7 04 25 00 01 40 00 40 00 40 00", // movq $0x400040, 0x400100
      "ff 14 25 00 01 40 00", // call *0x400100
      "c3", // ret
      ...Array.from({ length: 0x40 - 19 }, () => "90"), // nop padding out to the callee
      "b8 2a 00 00 00", // mov $42,%eax
      "c3", // ret
    ].join(" "),
  );

  it("calls the function the pointer names", () => {
    const { result } = run(program, {
      code: { bytes: program, virtualAddress: 0x400000n },
      indirectTargets: [0x400040n],
    });
    expect(result).toBe(0x2an);
  });

  it("traps when the pointer names something that is not a function this program translated", () => {
    // the same program with no target list: nothing to compare the pointer against, so the call is the trap a jump to garbage is
    expect(() => run(program, { code: { bytes: program, virtualAddress: 0x400000n } })).toThrow();
  });
});

describe("a switch compiled to a jump table", () => {
  /*
   * A `call` into a function whose body is the four instructions gcc emits for a `switch` over a dense range, and a table of offsets in
   * the image's data rather than its code:
   *
   *  16: mov $2,%edx                     the value switched on
   *  21: cmp $3,%edx; 24: ja default     the bounds check, which is also how the table's length is known
   *  26: lea table(%rip),%rcx            the table is at 0x400100
   *  33: movsxd (%rcx,%rdx,4),%rdx
   *  37: add %rcx,%rdx
   *  40: jmp *%rdx
   *  42: case 0 answers 10, 48: case 1 answers 11, 54: case 2 answers 12, 60: case 3 answers 13, 66: default answers 99
   */
  const code = (maskBounded = false): Uint8Array =>
    hex(
      "e8 0b 00 00 00 c3 90 90 90 90 90 90 90 90 90 90 " +
        "ba 02 00 00 00 " +
        // the bounds check, or the mask glibc's own helper uses instead: `and $0x3` is four entries and no `cmp` at all
        (maskBounded ? "83 e2 03 90 90" : "83 fa 03 77 28") +
        " 48 8d 0d df 00 00 00 48 63 14 91 48 01 ca ff e2 " +
        "b8 0a 00 00 00 c3 b8 0b 00 00 00 c3 b8 0c 00 00 00 c3 b8 0d 00 00 00 c3 b8 63 00 00 00 c3",
    );

  /** Runs the switch with the table at 0x400100, and with the image's data reader or without one. */
  const runSwitch = (readable: boolean, maskBounded = false): bigint => {
    const image = imageOf();
    const addresses = addressSpaceOf(image);
    const memory = new runtime.Memory({ initial: 1 });
    const buffer = new Uint8Array(memory.buffer);
    buffer.fill(0);
    // each entry is an offset from the table's own address, which is what makes the table position-independent
    const view = new DataView(memory.buffer);
    [42, 48, 54, 60].forEach((target, slot) => view.setInt32(0x100 + slot * 4, target - 0x100, true));
    const readData = (address: bigint, width: number): bigint | undefined => {
      const start = addresses.offsetOf(address);
      if (start === undefined || start + width > buffer.length) return undefined;
      let value = 0n;
      for (let byte = width - 1; byte >= 0; byte -= 1) value = (value << 8n) | BigInt(buffer[start + byte]!);
      return value;
    };
    const bytes = code(maskBounded);
    const module = translateBytes(bytes, "main", {
      addresses,
      memoryPages: 1,
      codeAddress: 0x400000n,
      code: { bytes, virtualAddress: 0x400000n },
      ...(readable ? { readData } : {}),
    });
    const instance = new runtime.Instance(new runtime.Module(module), { env: { memory } });
    return instance.exports.main!();
  };

  it("runs the case the table names, which no branch in the bytes names", () => {
    expect(runSwitch(true)).toBe(12n);
  });

  it("reads a table whose length is a mask rather than a comparison", () => {
    // `and $0x3,%edx` bounds the table the way a `cmp` does: four entries, and the case the value names runs
    expect(runSwitch(true, true)).toBe(12n);
  });

  it("refuses the case bodies when the table cannot be read, because nothing else reaches them", () => {
    expect(() => runSwitch(false)).toThrow(/nothing reaches the bytes/);
  });
});

describe("a switch table whose base is loaded far from the jump", () => {
  /*
   * The table's address is invariant, so gcc computes it once and keeps it: the `lea` can be twenty instructions from the jump, with
   * the value switched on read, the bounds check and other tests in between. The reader used to require the `lea` exactly three
   * instructions before the jump, which read only the gcc shape where the base is computed where it is used.
   *
   * Each fixture is a `call` into the function below, because a function the program calls is walked and the entry's own bytes are
   * not: what is being tested is the walk reading a table.
   *
   * The switch, at 0x10:
   *  10: mov $2,%edx                       the value switched on
   *  15: lea table(%rip),%rcx              the base, five instructions early
   *  1c: mov %edi,%edi ... 23: mov %r9d,%r9d     four instructions that leave %rcx alone
   *  26: cmp $3,%edx; 29: ja default
   *  2b: movsxd (%rcx,%rdx,4),%rdx; 2f: add %rcx,%rdx; 32: jmp *%rdx
   *  34: case 0 answers 10 ... 4c: default answers 99, and the table at 0x100 in the image's data
   */
  const distant = (): Uint8Array =>
    hex(
      "e8 0b 00 00 00 c3 90 90 90 90 90 90 90 90 90 90 " +
        "ba 02 00 00 00 48 8d 0d e4 00 00 00 89 ff 89 f6 45 89 c0 45 89 c9 83 fa 03 77 21 " +
        "48 63 14 91 48 01 ca ff e2 " +
        "b8 0a 00 00 00 c3 b8 0b 00 00 00 c3 b8 0c 00 00 00 c3 b8 0d 00 00 00 c3 b8 63 00 00 00 c3",
    );

  /*
   * The same table with a loop around it, which is what gcc emits for a `switch` inside a loop: the `jne` at 0x25 goes back to the
   * `lea` at 0x17, so it passes the write again and the base is still known at the load.
   *
   *  10: xor %edi,%edi; 12: mov $2,%edx; 17: lea table(%rip),%rcx; 1e: cmp $3,%edx; 21: ja default
   *  23: test %edi,%edi; 25: jne 17; 27: movsxd (%rcx,%rdx,4),%rdx; 2b: add %rcx,%rdx; 2e: jmp *%rdx
   *  30: case 0 answers 10 ... 48: default answers 99, and the table at 0x100
   */
  const looped = (): Uint8Array =>
    hex(
      "e8 0b 00 00 00 c3 90 90 90 90 90 90 90 90 90 90 " +
        "31 ff ba 02 00 00 00 48 8d 0d e2 00 00 00 83 fa 03 77 25 85 ff 75 f0 48 63 14 91 48 01 ca ff e2 " +
        "b8 0a 00 00 00 c3 b8 0b 00 00 00 c3 b8 0c 00 00 00 c3 b8 0d 00 00 00 c3 b8 63 00 00 00 c3",
    );

  /*
   * A call between the write and the load. The base is in `%rcx`, which a callee is free to clobber, so the register's value at the
   * load is not something these bytes say and the table is not read.
   *
   *  10: mov $2,%edx; 15: lea table(%rip),%rcx; 1c: call helper; 21: cmp $3,%edx; 24: ja default
   *  26: movsxd (%rcx,%rdx,4),%rdx; 2a: add %rcx,%rdx; 2d: jmp *%rdx
   *  2f: case 0 answers 10 ... 47: default answers 99; 4d: helper: ret
   */
  const clobbered = (): Uint8Array =>
    hex(
      "e8 0b 00 00 00 c3 90 90 90 90 90 90 90 90 90 90 " +
        "ba 02 00 00 00 48 8d 0d e4 00 00 00 e8 2c 00 00 00 83 fa 03 77 21 48 63 14 91 48 01 ca ff e2 " +
        "b8 0a 00 00 00 c3 b8 0b 00 00 00 c3 b8 0c 00 00 00 c3 b8 0d 00 00 00 c3 b8 63 00 00 00 c3 c3",
    );

  /** Runs one of the fixtures with its table written into the image's data at 0x100, which is where each `lea` names it. */
  const runWithTable = (bytes: Uint8Array, cases: readonly number[]): bigint => {
    const image = imageOf();
    const addresses = addressSpaceOf(image);
    const memory = new runtime.Memory({ initial: 1 });
    const buffer = new Uint8Array(memory.buffer);
    buffer.fill(0);
    // each entry is an offset from the table's own address, which is what makes the table position-independent
    const view = new DataView(memory.buffer);
    cases.forEach((target, slot) => view.setInt32(0x100 + slot * 4, target - 0x100, true));
    const readData = (address: bigint, width: number): bigint | undefined => {
      const start = addresses.offsetOf(address);
      if (start === undefined || start + width > buffer.length) return undefined;
      let value = 0n;
      for (let byte = width - 1; byte >= 0; byte -= 1) value = (value << 8n) | BigInt(buffer[start + byte]!);
      return value;
    };
    const module = translateBytes(bytes, "main", {
      addresses,
      memoryPages: 1,
      codeAddress: 0x400000n,
      code: { bytes, virtualAddress: 0x400000n },
      readData,
    });
    const instance = new runtime.Instance(new runtime.Module(module), { env: { memory } });
    return instance.exports.main!();
  };

  it("reads a table whose base was computed long before the jump", () => {
    expect(runWithTable(distant(), [0x34, 0x3a, 0x40, 0x46])).toBe(12n);
  });

  it("reads a table whose base is written before the loop the switch sits in", () => {
    expect(runWithTable(looped(), [0x30, 0x36, 0x3c, 0x42])).toBe(12n);
  });

  it("refuses it when a call sits between the write and the load, because a callee may clobber the base", () => {
    expect(() => runWithTable(clobbered(), [0x2f, 0x35, 0x3b, 0x41])).toThrow(/its table cannot be read/);
  });
});
