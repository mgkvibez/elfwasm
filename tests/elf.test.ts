import { describe, expect, it } from "vitest";

import { ElfFormatError, isElf64, loadableSegments, parseElfHeader, parseProgramHeaders } from "../src/elf.js";

/** A synthetic ELF64 header, so the parser is tested against bytes rather than a checked-in binary. */
function buildElf(options: { machine?: number; phnum?: number } = {}): Uint8Array {
  const bytes = new Uint8Array(64 + 56 * (options.phnum ?? 1));
  bytes.set([0x7f, 0x45, 0x4c, 0x46], 0);
  bytes[4] = 2;
  bytes[5] = 1;
  const dv = new DataView(bytes.buffer);
  dv.setUint16(16, 2, true);
  dv.setUint16(18, options.machine ?? 0x3e, true);
  dv.setBigUint64(24, 0x401000n, true);
  dv.setBigUint64(32, 64n, true);
  dv.setUint16(54, 56, true);
  dv.setUint16(56, options.phnum ?? 1, true);

  const ph = 64;
  dv.setUint32(ph, 1, true);
  dv.setUint32(ph + 4, 5, true);
  dv.setBigUint64(ph + 8, 0n, true);
  dv.setBigUint64(ph + 16, 0x400000n, true);
  dv.setBigUint64(ph + 32, 120n, true);
  dv.setBigUint64(ph + 40, 120n, true);
  return bytes;
}

describe("ELF64 header parsing", () => {
  it("reads entry point, machine and program header table", () => {
    const header = parseElfHeader(buildElf());
    expect(header.entry).toBe(0x401000n);
    expect(header.machine).toBe(0x3e);
    expect(header.programHeaderCount).toBe(1);
  });

  it("accepts aarch64 as well as x86-64", () => {
    expect(parseElfHeader(buildElf({ machine: 0xb7 })).machine).toBe(0xb7);
  });

  it("refuses a 32-bit binary by name instead of misreading it", () => {
    const bytes = buildElf();
    bytes[4] = 1;
    expect(() => parseElfHeader(bytes)).toThrow(/only ELF64 is supported/);
  });

  it("refuses a non-ELF file", () => {
    expect(isElf64(new Uint8Array(80))).toBe(false);
    expect(() => parseElfHeader(new Uint8Array(80))).toThrow(/bad magic/);
  });

  it("refuses a truncated header", () => {
    expect(() => parseElfHeader(buildElf().slice(0, 40))).toThrow(/truncated ELF header/);
  });

  it("refuses an unsupported machine rather than guessing", () => {
    expect(() => parseElfHeader(buildElf({ machine: 0x03 }))).toThrow(ElfFormatError);
  });

  it("returns only PT_LOAD segments for mapping", () => {
    const header = parseElfHeader(buildElf());
    const segments = loadableSegments(parseProgramHeaders(buildElf(), header));
    expect(segments).toHaveLength(1);
    expect(segments[0]?.virtualAddress).toBe(0x400000n);
  });

  it("rejects a program header table that runs past the file", () => {
    const header = parseElfHeader(buildElf());
    expect(() => parseProgramHeaders(buildElf().slice(0, 70), header)).toThrow(/runs past the end/);
  });
});
