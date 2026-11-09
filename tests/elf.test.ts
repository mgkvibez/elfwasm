import { describe, expect, it } from "vitest";

import {
  ElfFormatError,
  findSection,
  isElf64,
  loadableSegments,
  parseRelocations,
  parseElfHeader,
  parseProgramHeaders,
  parseSectionHeaders,
  parseSymbols,
} from "../src/elf.js";

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

/**
 * A synthetic ELF64 with a section table, a `.shstrtab`, a `.symtab` and a `.strtab`.
 *
 * Sections: 0 null, 1 .text, 2 .shstrtab, 3 .symtab, 4 .strtab
 * Symbols:  index 0 null, 1 main (FUNC, GLOBAL), 2 helper (FUNC, LOCAL)
 */
function buildElfWithSections(options: { shnum?: number; shstrndx?: number; symEntrySize?: number } = {}): Uint8Array {
  const names = "\u0000.text\u0000.shstrtab\u0000.symtab\u0000.strtab\u0000";
  const strings = "\u0000main\u0000helper\u0000";
  const shnum = options.shnum ?? 5;
  const shstrndx = options.shstrndx ?? 2;
  const symEntrySize = options.symEntrySize ?? 24;

  const shstrOffset = 64 + 56; // header + one program header
  const shstrBytes = names.length;
  const symOffset = shstrOffset + shstrBytes;
  const symBytes = 24 * 3;
  const strOffset = symOffset + symBytes;
  const strBytes = strings.length;
  const shoff = strOffset + strBytes;

  // always lay out all five sections; `shnum` only changes what the header claims
  const bytes = new Uint8Array(shoff + 64 * Math.max(shnum, 5));
  const dv = new DataView(bytes.buffer);
  bytes.set([0x7f, 0x45, 0x4c, 0x46], 0);
  bytes[4] = 2;
  bytes[5] = 1;
  dv.setUint16(16, 2, true); // ET_EXEC
  dv.setUint16(18, 0x3e, true); // x86-64
  dv.setBigUint64(24, 0x401000n, true);
  dv.setBigUint64(32, 64n, true);
  dv.setUint16(54, 56, true);
  dv.setUint16(56, 1, true);
  dv.setBigUint64(40, BigInt(shoff), true);
  dv.setUint16(58, 64, true);
  dv.setUint16(60, shnum, true);
  dv.setUint16(62, shstrndx, true);

  // program header: one PT_LOAD
  dv.setUint32(64, 1, true);
  dv.setUint32(64 + 4, 5, true);
  dv.setBigUint64(64 + 32, 120n, true);
  dv.setBigUint64(64 + 40, 120n, true);

  for (let i = 0; i < names.length; i += 1) bytes[shstrOffset + i] = names.charCodeAt(i);
  for (let i = 0; i < strings.length; i += 1) bytes[strOffset + i] = strings.charCodeAt(i);

  const symAt = (i: number, nameOffset: number, info: number, value: bigint, size: bigint) => {
    const base = symOffset + i * 24;
    dv.setUint32(base, nameOffset, true);
    dv.setUint8(base + 4, info);
    dv.setUint16(base + 6, 1, true);
    dv.setBigUint64(base + 8, value, true);
    dv.setBigUint64(base + 16, size, true);
  };
  symAt(0, 0, 0, 0n, 0n); // null symbol
  symAt(1, 1, 0x12, 0x401000n, 32n); // GLOBAL FUNC
  symAt(2, 6, 0x02, 0x401020n, 16n); // LOCAL FUNC

  const section = (i: number, nameOff: number, type: number, offset: number, size: number, link: number, entsize: number) => {
    const base = shoff + i * 64;
    dv.setUint32(base, nameOff, true);
    dv.setUint32(base + 4, type, true);
    dv.setBigUint64(base + 8, type === 2 || type === 11 ? 2n : 0n, true);
    dv.setBigUint64(base + 24, BigInt(offset), true);
    dv.setBigUint64(base + 32, BigInt(size), true);
    dv.setUint32(base + 40, link, true);
    dv.setBigUint64(base + 56, BigInt(entsize), true);
  };
  section(0, 0, 0, 0, 0, 0, 0); // null
  section(1, 1, 1, 0, 0, 0, 0); // .text, SHT_PROGBITS
  section(2, 7, 3, shstrOffset, shstrBytes, 0, 0); // .shstrtab
  section(3, 17, 2, symOffset, symBytes, 4, symEntrySize); // .symtab -> .strtab
  section(4, 25, 3, strOffset, strBytes, 0, 0); // .strtab

  return bytes;
}

/**
 * A synthetic ELF64 with sections, symbols and optional RELA relocations.
 *
 * Sections: 0 null, 1 .text, 2 .shstrtab, 3 .symtab, 4 .strtab, 5 .rela.text (when asked for)
 */
function sectionElf(
  options: {
    relas?: { target: bigint; symbol: number; type: number; addend: bigint }[];
    relStyle?: boolean;
    relaEntrySize?: number;
  } = {},
): Uint8Array {
  const relas = options.relas ?? [];
  const hasRelocations = relas.length > 0 || options.relStyle === true;
  const shnum = hasRelocations ? 6 : 5;
  const names = hasRelocations
    ? "\u0000.text\u0000.shstrtab\u0000.symtab\u0000.strtab\u0000.rela.text\u0000"
    : "\u0000.text\u0000.shstrtab\u0000.symtab\u0000.strtab\u0000";
  const strings = "\u0000main\u0000helper\u0000";

  const shstrOffset = 64 + 56;
  const symOffset = shstrOffset + names.length;
  const strOffset = symOffset + 24 * 3;
  const relaOffset = strOffset + strings.length;
  const relaCount = options.relStyle ? 1 : relas.length;
  const relaEntrySize = options.relaEntrySize ?? 24;
  const relaBytes = options.relStyle ? 16 : relaCount * 24;
  const shoff = relaOffset + relaBytes;

  const bytes = new Uint8Array(shoff + 64 * shnum);
  const dv = new DataView(bytes.buffer);
  bytes.set([0x7f, 0x45, 0x4c, 0x46], 0);
  bytes[4] = 2;
  bytes[5] = 1;
  dv.setUint16(16, 2, true);
  dv.setUint16(18, 0x3e, true);
  dv.setBigUint64(24, 0x401000n, true);
  dv.setBigUint64(32, 64n, true);
  dv.setUint16(54, 56, true);
  dv.setUint16(56, 1, true);
  dv.setBigUint64(40, BigInt(shoff), true);
  dv.setUint16(58, 64, true);
  dv.setUint16(60, shnum, true);
  dv.setUint16(62, 2, true);
  dv.setUint32(64, 1, true);
  dv.setUint32(64 + 4, 5, true);
  dv.setBigUint64(64 + 32, 120n, true);
  dv.setBigUint64(64 + 40, 120n, true);

  for (let i = 0; i < names.length; i += 1) bytes[shstrOffset + i] = names.charCodeAt(i);
  for (let i = 0; i < strings.length; i += 1) bytes[strOffset + i] = strings.charCodeAt(i);

  const symAt = (i: number, nameOffset: number, info: number, value: bigint, size: bigint) => {
    const base = symOffset + i * 24;
    dv.setUint32(base, nameOffset, true);
    dv.setUint8(base + 4, info);
    dv.setUint16(base + 6, 1, true);
    dv.setBigUint64(base + 8, value, true);
    dv.setBigUint64(base + 16, size, true);
  };
  symAt(0, 0, 0, 0n, 0n);
  symAt(1, 1, 0x12, 0x401000n, 32n);
  symAt(2, 6, 0x02, 0x401020n, 16n);

  const section = (i: number, nameOff: number, type: number, offset: number, size: number, link: number, entsize: number) => {
    const base = shoff + i * 64;
    dv.setUint32(base, nameOff, true);
    dv.setUint32(base + 4, type, true);
    dv.setBigUint64(base + 24, BigInt(offset), true);
    dv.setBigUint64(base + 32, BigInt(size), true);
    dv.setUint32(base + 40, link, true);
    dv.setBigUint64(base + 56, BigInt(entsize), true);
  };
  section(0, 0, 0, 0, 0, 0, 0);
  section(1, 1, 1, 0, 0, 0, 0);
  section(2, 7, 3, shstrOffset, names.length, 0, 0);
  section(3, 17, 2, symOffset, 24 * 3, 4, 24);
  section(4, 25, 3, strOffset, strings.length, 0, 0);
  if (hasRelocations) {
    section(5, 35, options.relStyle ? 9 : 4, relaOffset, relaBytes, 3, relaEntrySize);
  }

  relas.forEach((rela, index) => {
    const base = relaOffset + index * 24;
    dv.setBigUint64(base, rela.target, true);
    dv.setBigUint64(base + 8, (BigInt(rela.symbol) << 32n) | BigInt(rela.type), true);
    dv.setBigInt64(base + 16, rela.addend, true);
  });

  return bytes;
}

describe("ELF64 section headers", () => {
  it("reads the table and resolves every section name", () => {
    const header = parseElfHeader(buildElfWithSections());
    const sections = parseSectionHeaders(buildElfWithSections(), header);
    expect(sections.map((s) => s.name)).toEqual(["", ".text", ".shstrtab", ".symtab", ".strtab"]);
    expect(findSection(sections, ".symtab")?.type).toBe(2);
  });

  it("refuses a stripped object instead of pretending it has sections", () => {
    const bytes = buildElfWithSections({ shnum: 0 });
    const header = parseElfHeader(bytes);
    expect(() => parseSectionHeaders(bytes, header)).toThrow(/no section headers/);
  });

  it("refuses a name table index past the end of the table", () => {
    const bytes = buildElfWithSections({ shstrndx: 9 });
    const header = parseElfHeader(bytes);
    expect(() => parseSectionHeaders(bytes, header)).toThrow(/past the 5 sections/);
  });

  it("refuses a section table that runs past the end of the file", () => {
    const full = buildElfWithSections();
    const truncated = full.subarray(0, full.length - 64);
    const header = parseElfHeader(full);
    expect(() => parseSectionHeaders(truncated, header)).toThrow(/runs past the end of the file/);
  });
});

describe("ELF64 symbol table", () => {
  it("reads names, values, sizes, binding and type", () => {
    const bytes = buildElfWithSections();
    const sections = parseSectionHeaders(bytes, parseElfHeader(bytes));
    const symbols = parseSymbols(bytes, sections);
    expect(symbols).toHaveLength(2); // the reserved null symbol is skipped
    expect(symbols[0]).toMatchObject({ name: "main", bind: 1, type: 2, value: 0x401000n, size: 32n });
    expect(symbols[1]).toMatchObject({ name: "helper", bind: 0, type: 2 });
  });

  it("refuses a binary with no .symtab instead of returning an empty list", () => {
    const bytes = buildElfWithSections();
    const sections = parseSectionHeaders(bytes, parseElfHeader(bytes)).filter((s) => s.type !== 2);
    expect(() => parseSymbols(bytes, sections)).toThrow(/no \.symtab section/);
  });

  it("refuses a symbol table whose entries are not Elf64_Sym sized", () => {
    const bytes = buildElfWithSections({ symEntrySize: 16 });
    const sections = parseSectionHeaders(bytes, parseElfHeader(bytes));
    expect(() => parseSymbols(bytes, sections)).toThrow(/entry size 16 is not 24/);
  });

  it("refuses .dynsym when the object has only static symbols", () => {
    const bytes = buildElfWithSections();
    const sections = parseSectionHeaders(bytes, parseElfHeader(bytes));
    expect(() => parseSymbols(bytes, sections, { dynamic: true })).toThrow(/no \.dynsym section/);
  });
});

/**
 * A real binary, when the machine has one.
 *
 * Synthetic tables prove the parser matches the spec as I read it; this proves it matches files a linker actually
 * produced, which is the only claim that matters. The host read is a dynamic import so the parser keeps zero
 * dependencies and no Node types, and the test is skipped rather than failed where no ELF64 host binary exists.
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

  it("parses header, segments, sections and dynamic symbols end to end", async () => {
    let bytes: Uint8Array | undefined;
    for (const candidate of ["/bin/ls", "/usr/bin/ls", "/bin/sh"]) {
      const loaded = await load(candidate);
      if (loaded !== undefined && isElf64(loaded)) {
        bytes = loaded;
        break;
      }
    }
    if (bytes === undefined) return; // no ELF64 host binary here: nothing to assert

    const header = parseElfHeader(bytes);
    const segments = parseProgramHeaders(bytes, header);
    const sectionsForRelocations = parseSectionHeaders(bytes, header);
    const relocations = parseRelocations(bytes, sectionsForRelocations, { dynamic: true });
    // a dynamically linked binary relocates something; zero would mean we silently read nothing
    expect(relocations.length).toBeGreaterThan(0);
    expect(relocations.some((relocation) => relocation.symbolName !== "")).toBe(true);
    const sections = parseSectionHeaders(bytes, header);

    expect(loadableSegments(segments).length).toBeGreaterThan(0);
    expect(sections.length).toBe(header.sectionHeaderCount);
    expect(findSection(sections, ".text")).toBeDefined();

    const dynamic = parseSymbols(bytes, sections, { dynamic: true });
    expect(dynamic.length).toBeGreaterThan(0);
    expect(dynamic.some((symbol) => symbol.name.length > 0)).toBe(true);
    for (const segment of loadableSegments(segments)) {
      expect(Number(segment.offset + segment.fileSize)).toBeLessThanOrEqual(bytes.length);
    }
  });
});

describe("ELF64 relocations", () => {
  it("reads a R_X86_64_64 relocation with its symbol and addend", () => {
    const bytes = sectionElf({ relas: [{ target: 0x404000n, symbol: 1, type: 1, addend: 8n }] });
    const header = parseElfHeader(bytes);
    const sections = parseSectionHeaders(bytes, header);
    const relocations = parseRelocations(bytes, sections);
    expect(relocations).toHaveLength(1);
    expect(relocations[0]!.offset).toBe(0x404000n);
    expect(relocations[0]!.type).toBe(1);
    expect(relocations[0]!.addend).toBe(8n);
    expect(relocations[0]!.symbolName).toBe("main");
  });

  it("keeps a negative addend negative", () => {
    const bytes = sectionElf({ relas: [{ target: 0x1000n, symbol: 2, type: 4, addend: -16n }] });
    const header = parseElfHeader(bytes);
    const relocations = parseRelocations(bytes, parseSectionHeaders(bytes, header));
    expect(relocations[0]!.addend).toBe(-16n);
    expect(relocations[0]!.symbolName).toBe("helper");
  });

  it("treats symbol index zero as no symbol rather than the null entry", () => {
    const bytes = sectionElf({ relas: [{ target: 0x2000n, symbol: 0, type: 8, addend: 0n }] });
    const header = parseElfHeader(bytes);
    const relocations = parseRelocations(bytes, parseSectionHeaders(bytes, header));
    expect(relocations[0]!.symbolIndex).toBe(0);
    expect(relocations[0]!.symbolName).toBe("");
  });

  it("refuses REL relocations by name instead of reading addend zero", () => {
    const bytes = sectionElf({ relStyle: true });
    const header = parseElfHeader(bytes);
    expect(() => parseRelocations(bytes, parseSectionHeaders(bytes, header))).toThrow(/only RELA relocations are supported/);
  });

  it("refuses an object with no relocations at all", () => {
    const bytes = sectionElf({});
    const header = parseElfHeader(bytes);
    expect(() => parseRelocations(bytes, parseSectionHeaders(bytes, header))).toThrow(/no relocations in this object/);
  });

  it("refuses a relocation entry size that is not Elf64_Rela", () => {
    const bytes = sectionElf({ relas: [{ target: 1n, symbol: 1, type: 1, addend: 0n }], relaEntrySize: 16 });
    const header = parseElfHeader(bytes);
    expect(() => parseRelocations(bytes, parseSectionHeaders(bytes, header))).toThrow(/expected 24/);
  });
});
