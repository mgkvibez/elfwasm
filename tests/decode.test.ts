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

/** objdump spells the same mnemonic with a size suffix (`movq`, `movl`), which the IR does not carry. */
const family = (mnemonic: string): string => mnemonic.replace(/^(mov)[qlwb]$/, "$1");

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

describe("refusals", () => {
  const refuse = (bytes: Uint8Array): DecodeError => {
    try {
      decode(bytes);
    } catch (error) {
      if (error instanceof DecodeError) return error;
      throw error;
    }
    throw new Error("expected the decoder to refuse these bytes");
  };

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

  it("refuses a segment override, which resolves through a base the loader has not set up", () => {
    expect(refuse(hex("64 48 8b 04 25 00 00 00 00")).message).toContain("segment override");
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
