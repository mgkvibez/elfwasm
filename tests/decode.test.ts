import { describe, expect, it } from "vitest";

import { DecodeError, decode, formatInstruction, type Instruction, type Operand } from "../src/decode.js";
import listing from "./fixtures/hello.dis.json" with { type: "json" };
import { findSection, isElf64, parseElfHeader, parseProgramHeaders, parseSectionHeaders } from "../src/elf.js";

/**
 * The fixture is a real assembly file, assembled by `as` and disassembled by `objdump` through
 * tools/build-decode-fixture.sh. The expectation is therefore binutils' opinion of where each instruction begins, which
 * is the thing a decoder can quietly get wrong: a wrong length still yields a plausible looking instruction, just not
 * the one the compiler emitted, and every instruction after it is out of phase.
 */
const hex = (text: string): Uint8Array =>
  Uint8Array.from(text.trim().split(/\s+/).map((pair) => Number.parseInt(pair, 16)));

/**
 * objdump's spelling mapped onto the IR's, in one place so the mapping is auditable.
 *
 * Two different things are going on. Size suffixes (`movq`, `movl`, `movw`, `movb`) say how wide the operand is, which the
 * IR carries on the operand instead of the mnemonic, so they are stripped. Aliases are objdump's own names for
 * instructions with a longer official name (`cltq` is `cdqe`, `cqto` is `cqo`), and the extend instructions carry the
 * source size and the destination size in the mnemonic (`movzbl` is "zero-extend byte to long"), which the IR spells as
 * `movzx` with a byte source operand.
 */
const ALIASES: Record<string, string> = {
  cltq: "cdqe",
  // objdump calls the 0x63 instruction movslq, and the IR keeps Intel's movsxd: same opcode, same operand
  movslq: "movsxd",
  cwtl: "cwde",
  cqto: "cqo",
  cltd: "cdq",
};
const family = (mnemonic: string): string => {
  if (ALIASES[mnemonic] !== undefined) return ALIASES[mnemonic]!;
  // movzbl / movzwl / movsbq / movswq: the sizes are in the name, the destination size is the operand's `bits`
  const extended = /^(movz|movs)(b|w)([wlq])$/.exec(mnemonic);
  if (extended !== null) return `${extended[1]}x`;
  return mnemonic.replace(/^(mov|test|imul|neg|not|shl|shr|sar|cmp|add|sub|and|or|xor)[qlwb]$/, "$1");
};

const asBytes = async (): Promise<Uint8Array> => {
  // @ts-expect-error node:fs is intentionally outside this project's type surface
  const { readFileSync } = await import("node:fs");
  // vitest runs from the repository root, and the fixture is committed, so the test needs no binutils of its own
  return new Uint8Array(readFileSync("tests/fixtures/hello.text.bin"));
};

const operandsOf = (instruction: Instruction): readonly Operand[] => instruction.operands;

describe("decoding assembler output", () => {
  it("finds every instruction at the offset objdump found it, and none in between", async () => {
    const instructions = decode(await asBytes());
    expect(instructions).toHaveLength(listing.length);
    for (const [index, expected] of listing.entries()) {
      const actual = instructions[index]!;
      expect({ offset: actual.offset, length: actual.length }).toEqual({
        offset: expected.offset,
        length: expected.length,
      });
      expect(family(expected.mnemonic)).toBe(actual.op);
    }
  });

  it("accounts for every byte exactly once", async () => {
    const bytes = await asBytes();
    const instructions = decode(bytes);
    let cursor = 0;
    for (const instruction of instructions) {
      expect(instruction.offset).toBe(cursor);
      cursor += instruction.length;
    }
    expect(cursor).toBe(bytes.length);
  });

  it("reads a RIP-relative lea as a displacement from the instruction, not as an absolute address", async () => {
    const instructions = decode(await asBytes());
    const lea = instructions.find((instruction) => instruction.op === "lea")!;
    expect(formatInstruction(lea)).toBe("lea rsi [rip]");
    const [destination, source] = operandsOf(lea);
    expect(destination).toMatchObject({ kind: "reg", name: "rsi", bits: 64 });
    expect(source).toMatchObject({ kind: "memory", ripRelative: true, displacement: 0n });
  });

  it("reads base, index and scale out of a SIB byte", async () => {
    const instructions = decode(await asBytes());
    const load = instructions.find(
      (instruction) =>
        instruction.op === "mov" &&
        instruction.operands[0]?.kind === "reg" &&
        instruction.operands[0].name === "rcx",
    )!;
    expect(formatInstruction(load)).toBe("mov rcx [rsp+4*rax+0x8]");
    expect(load.operands[1]).toMatchObject({
      kind: "memory",
      base: "rsp",
      index: "rax",
      scale: 4,
      displacement: 8n,
    });
  });

  it("sign-extends a negative displacement, and a displacement small enough for one byte", async () => {
    const instructions = decode(await asBytes());
    const local = instructions.find((instruction) => formatInstruction(instruction) === "mov [rbp-0x8] 0")!;
    expect(local.operands[0]).toMatchObject({ kind: "memory", base: "rbp", displacement: -8n });
    expect(local.length).toBe(8);
    const subtraction = instructions.find((instruction) => formatInstruction(instruction) === "sub rsp 16")!;
    expect(subtraction.operands[1]).toEqual({ kind: "immediate", value: 16n });
    expect(subtraction.length).toBe(4);
  });

  it("sign-extends a byte-sized immediate, which encodes signed and must not come back as 253", () => {
    // 48 83 e9 fd is `sub rcx, -3`
    const [negative] = decode(hex("48 83 e9 fd"));
    expect(negative!.op).toBe("sub");
    expect(formatInstruction(negative!)).toBe("sub rcx -3");
    expect(negative!.operands[1]).toEqual({ kind: "immediate", value: -3n });
  });

  it("reads the system call, which is two bytes and nothing else", async () => {
    const instructions = decode(await asBytes());
    const syscall = instructions.findIndex((instruction) => instruction.op === "syscall");
    expect(syscall).toBeGreaterThan(0);
    expect(instructions[syscall]!.length).toBe(2);
    expect(instructions[syscall]!.operands).toEqual([]);
  });

  it("accepts both encodings of a conditional jump, the one-byte and the two-byte form", () => {
    // 75 fe = jne -2 (the short form a loop body uses), 0f 85 f6 ff ff ff = jne -10 (the long form)
    const short = decode(hex("75 fe"));
    const long = decode(hex("0f 85 f6 ff ff ff"));
    expect(short[0]).toMatchObject({ op: "jne", length: 2 });
    expect(short[0]!.operands[0]).toEqual({ kind: "immediate", value: -2n });
    expect(long[0]).toMatchObject({ op: "jne", length: 6 });
    expect(long[0]!.operands[0]).toEqual({ kind: "immediate", value: -10n });
  });
});

describe("the pieces a compiler leans on", () => {
  it("names a byte register from the encoding: ah without REX, spl with it", () => {
    // 88 /r is `mov r/m8, r8`; with rm 4 the name is `ah` unless REX is present, when it is `spl`
    expect(decode(hex("88 c4"))[0]!.operands).toEqual([
      { kind: "reg", name: "ah", bits: 8 },
      { kind: "reg", name: "al", bits: 8 },
    ]);
    expect(decode(hex("40 88 c4"))[0]!.operands).toEqual([
      { kind: "reg", name: "spl", bits: 8 },
      { kind: "reg", name: "al", bits: 8 },
    ]);
  });

  it("reads endbr64 as its own four-byte instruction rather than as a repeat prefix", () => {
    const [instruction, next] = decode(hex("f3 0f 1e fa 90"));
    expect({ op: instruction!.op, length: instruction!.length }).toEqual({ op: "endbr64", length: 4 });
    expect(next!.op).toBe("nop");
  });

  it("reads the linker's alignment padding as one nop, not as a 16-bit prefix", () => {
    const [instruction] = decode(hex("66 2e 0f 1f 84 00 00 00 00 00"));
    expect({ op: instruction!.op, length: instruction!.length }).toEqual({ op: "nop", length: 10 });
  });

  it("sign-extends and zero-extends from memory, with the destination width on the operand", () => {
    const [zero] = decode(hex("0f b6 07"));
    // a 32-bit operand keeps the 64-bit name and carries its width, which is the IR's convention for every width but
    // the byte, where the encoding itself distinguishes al from ah
    expect({ op: zero!.op, dst: zero!.operands[0] }).toEqual({ op: "movzx", dst: { kind: "reg", name: "rax", bits: 32 } });
    const [sign] = decode(hex("48 0f be 07"));
    expect({ op: sign!.op, dst: sign!.operands[0] }).toEqual({ op: "movsx", dst: { kind: "reg", name: "rax", bits: 64 } });
  });

  it("refuses a 16-bit register source for movzx, which would need the prefix it does not decode", () => {
    expect(refuseBytes(hex("0f b7 c0")).message).toContain("needs a 0x66 prefix");
  });

  it("calls an SSE encoding SSE, not a 16-bit operand size, when the 0x66 prefix is the mandatory one", () => {
    // 66 0f 6f is movdqa: the 0x66 here is part of the SIMD encoding, and saying "16-bit operand" would be wrong
    expect(refuseBytes(hex("66 0f 6f 05 00 00 00 00")).message).toContain("SSE");
    // and the unprefixed form, xorps, is SSE too
    expect(refuseBytes(hex("0f 57 c0")).message).toContain("SSE");
  });

  it("refuses mul, div and the rotates by name rather than pretending they are in the subset", () => {
    expect(refuseBytes(hex("f7 f1")).message).toContain("implicit registers");
    expect(refuseBytes(hex("d1 c0")).message).toContain("rotate");
  });

  it("reads a shift by cl as cl, not as a fixed count", () => {
    const [instruction] = decode(hex("48 d3 fa"));
    expect(instruction!.op).toBe("sar");
    expect(instruction!.operands[1]).toEqual({ kind: "reg", name: "cl", bits: 8 });
  });

  it("reads the sign-extension pair by width, so cdqe and cqo are not confused with cwde and cdq", () => {
    expect(decode(hex("48 98"))[0]!.op).toBe("cdqe");
    expect(decode(hex("98"))[0]!.op).toBe("cwde");
    expect(decode(hex("48 99"))[0]!.op).toBe("cqo");
    expect(decode(hex("99"))[0]!.op).toBe("cdq");
  });
});

/** The error a refusal raises, so a test can assert on what it says rather than only that it threw. */
const refuseBytes = (bytes: Uint8Array): DecodeError => {
  try {
    decode(bytes);
  } catch (error) {
    if (error instanceof DecodeError) return error;
    throw error;
  }
  throw new Error(`the decoder accepted these bytes: ${[...bytes].map((byte) => byte.toString(16)).join(" ")}`);
};

describe("refusals", () => {
  const refuse = refuseBytes;

  it("names the offset of an opcode it does not implement instead of skipping it", () => {
    // nop, then 0f 28 c0 (movaps) which needs SSE and is not in this subset, then ret
    const error = refuse(hex("90 0f 28 c0 c3"));
    expect(error.offset).toBe(1);
    expect(error.message).toContain("0x0f 0x28");
  });

  it("refuses a truncated instruction rather than reading past the buffer", () => {
    const error = refuse(hex("48 8b"));
    expect(error.message).toContain("past the end");
  });

  it("refuses a 16-bit operand size, which has no WASM local here", () => {
    expect(refuse(hex("66 89 c0")).message).toContain("16-bit");
  });

  it("reads the stack canary load with its segment, so `fs:0x28` is not left looking like address 0x28", () => {
    // 64 48 8b 04 25 28 00 00 00 is `mov %fs:0x28, %rax`, the first thing most functions do
    const [instruction] = decode(hex("64 48 8b 04 25 28 00 00 00"));
    expect(instruction!.op).toBe("mov");
    expect(instruction!.length).toBe(9);
    const source = instruction!.operands[1]!;
    expect(source).toMatchObject({ kind: "memory", segment: "fs", displacement: 0x28n, ripRelative: false });
  });

  it("refuses a lock prefix rather than dropping it, since dropping it changes the meaning", () => {
    expect(refuse(hex("f0 48 01 08")).message).toContain("lock or repeat");
  });

  it("refuses the lea form that cannot exist", () => {
    // 8d /r with mod 11: there is no register form of lea, so the bytes are not what they claim
    expect(refuse(hex("48 8d c0")).message).toContain("cannot be encoded");
  });
});

/**
 * A real compiler's output, when the machine has one.
 *
 * The fixture proves the decoder agrees with binutils on the subset. This proves the refusals are real: a system binary
 * uses instructions outside the subset, and the decoder has to stop exactly where they begin rather than sliding out of
 * phase. Before the refusal point every byte must be accounted for, which is the claim that matters.
 */
describe("a real ELF64 binary from the host", () => {
  const load = async (path: string): Promise<Uint8Array | undefined> => {
    try {
      // @ts-expect-error node:fs is intentionally outside this project's type surface
      const { readFileSync } = await import("node:fs");
      return new Uint8Array(readFileSync(path));
    } catch {
      return undefined;
    }
  };

  it("accounts for every byte up to the first instruction outside the subset, then refuses by offset", async () => {
    let bytes: Uint8Array | undefined;
    for (const candidate of ["/bin/ls", "/usr/bin/ls", "/bin/sh"]) {
      const loaded = await load(candidate);
      if (loaded !== undefined && isElf64(loaded)) {
        bytes = loaded;
        break;
      }
    }
    if (bytes === undefined) return;

    const header = parseElfHeader(bytes);
    const segments = parseProgramHeaders(bytes, header);
    const text = findSection(parseSectionHeaders(bytes, header), ".text");
    expect(text).toBeDefined();
    const code = bytes.subarray(Number(text!.offset), Number(text!.offset + text!.size));

    let decoded = 0;
    let cursor = 0;
    for (;;) {
      try {
        const instructions = decode(code.subarray(0, code.length));
        decoded = instructions.length;
        cursor = code.length;
        break;
      } catch (error) {
        if (!(error instanceof DecodeError)) throw error;
        // everything before the refusal must have been decoded, and it must be the refusal that stops us
        expect(error.offset).toBeGreaterThanOrEqual(0);
        if (error.offset === 0) {
          decoded = 0;
          cursor = 0;
          break;
        }
        // decode the prefix that is inside the subset: no byte may be skipped to reach the refusal point
        const prefix = code.subarray(0, error.offset);
        let prefixCursor = 0;
        while (prefixCursor < prefix.length) {
          const [instruction] = decode(prefix.subarray(prefixCursor));
          prefixCursor += instruction!.length;
          decoded += 1;
        }
        cursor = error.offset;
        break;
      }
    }
    expect(cursor).toBeLessThanOrEqual(code.length);
    expect(decoded).toBeGreaterThan(0);
    expect(segments.length).toBeGreaterThan(0);
  });
});

describe("group 3 at a byte's width", () => {
  it("reads test, not and neg, and names the encodings it does not have", () => {
    // f6 c6 48 is `test $0x48, %dh`, which is what the second instruction of a glibc ifunc resolver is
    const test = decode(hex("f6 c6 48"))[0]!;
    expect(test.op).toBe("test");
    expect(test.operands[0]).toMatchObject({ kind: "reg", name: "dh", bits: 8 });
    expect(test.operands[1]).toMatchObject({ kind: "immediate", value: 0x48n });
    expect(decode(hex("f6 d6"))[0]!.op).toBe("not");
    expect(decode(hex("f6 da"))[0]!.op).toBe("neg");
    // the register field selects the operation, so an undefined one is refused rather than read as an operand
    expect(() => decode(hex("f6 c8"))).toThrow(/0xf6 \/1 is undefined/);
    expect(() => decode(hex("f6 e0"))).toThrow(/mul, imul, div or idiv/);
  });

  it("reads mov r8, imm8 into a high byte, and into a low byte from the same code under REX", () => {
    // b6 is code 6 of 0xb0: `%dh` without REX, and `%sil` with one, because the four high bytes cannot be named under REX
    const high = decode(hex("b6 48"))[0]!;
    expect(high.op).toBe("mov");
    expect(high.operands[0]).toMatchObject({ kind: "reg", name: "dh", bits: 8 });
    const low = decode(hex("40 b6 48"))[0]!;
    expect(low.operands[0]).toMatchObject({ kind: "reg", name: "sil", bits: 8 });
  });
});

describe("a conditional move", () => {
  it("names the condition, keeps the destination in the modrm field and reads the width from REX", () => {
    // 0f 44 c3 is `cmove %ebx, %eax`: the destination is the register field, the source the r/m field
    const equal = decode(hex("0f 44 c3"))[0]!;
    expect(equal.op).toBe("cmove");
    // a 32-bit register keeps its 64-bit name and carries the width instead, which is how the rest of the IR spells it
    expect(equal.operands[0]).toMatchObject({ kind: "reg", name: "rax", bits: 32 });
    expect(equal.operands[1]).toMatchObject({ kind: "reg", name: "rbx", bits: 32 });
    // 0f 4f with REX.W is `cmovg %rbx, %rax`, and the condition is the low nibble of the opcode
    const greater = decode(hex("48 0f 4f c3"))[0]!;
    expect(greater.op).toBe("cmovg");
    expect(greater.operands[0]).toMatchObject({ kind: "reg", name: "rax", bits: 64 });
    // the source may be memory, which is loaded at the destination's width
    const fromMemory = decode(hex("0f 4c 03"))[0]!;
    expect(fromMemory.op).toBe("cmovl");
    expect(fromMemory.operands[1]).toMatchObject({ kind: "memory", bits: 32 });
  });
});
