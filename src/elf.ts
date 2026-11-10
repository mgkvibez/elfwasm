/**
 * ELF64 reading, which is the first real step of the translator.
 *
 * Nothing here executes anything. A translator that misreads the header produces confident nonsense downstream, so the
 * header is parsed into an explicit type once and every later stage reads that type instead of re-slicing bytes.
 *
 * Scope is deliberately narrow: little-endian ELF64, which is what static musl binaries for x86-64 and aarch64 are.
 * Big-endian, ELF32 and relocatable objects are refused by name rather than silently misparsed.
 */

export type ElfClass = 1 | 2;
export type ElfMachine = 0x3e | 0xb7;
export type ElfType = 2 | 3;

export interface ElfHeader {
  readonly elfClass: ElfClass;
  readonly type: ElfType;
  readonly machine: ElfMachine;
  readonly entry: bigint;
  readonly programHeaderOffset: bigint;
  readonly programHeaderCount: number;
  readonly programHeaderSize: number;
  readonly sectionHeaderOffset: bigint;
  readonly sectionHeaderCount: number;
  readonly sectionHeaderSize: number;
  readonly sectionNameIndex: number;
}

/** One entry of the section header table, with its name already resolved. */
export interface SectionHeader {
  readonly index: number;
  readonly name: string;
  readonly type: number;
  readonly flags: bigint;
  readonly address: bigint;
  readonly offset: bigint;
  readonly size: bigint;
  readonly link: number;
  readonly entrySize: bigint;
}

/** One entry of a symbol table, flattened from `Elf64_Sym`'s packed `st_info` byte. */
export interface ElfSymbol {
  readonly name: string;
  readonly value: bigint;
  readonly size: bigint;
  /** STB_LOCAL = 0, STB_GLOBAL = 1, STB_WEAK = 2 */
  readonly bind: number;
  /** STT_NOTYPE = 0, STT_OBJECT = 1, STT_FUNC = 2, STT_SECTION = 3, STT_FILE = 4 */
  readonly type: number;
  readonly sectionIndex: number;
}

export interface ProgramHeader {
  readonly type: number;
  readonly flags: number;
  readonly offset: bigint;
  readonly virtualAddress: bigint;
  readonly fileSize: bigint;
  readonly memorySize: bigint;
  /** `p_align`, which requires `p_vaddr` and `p_offset` to be congruent modulo it. */
  readonly alignment: bigint;
}

const ELF_MAGIC = [0x7f, 0x45, 0x4c, 0x46] as const;
const EHDR_SIZE = 64;
const PHDR_SIZE = 56;
const SHDR_SIZE = 64;
const SYM_SIZE = 24;
const PT_LOAD = 1;
const SHT_SYMTAB = 2;
const SHT_STRTAB = 3;
const SHT_DYNSYM = 11;

export class ElfFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ElfFormatError";
  }
}

function view(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

export function isElf64(bytes: Uint8Array): boolean {
  if (bytes.length < EHDR_SIZE) return false;
  if (!ELF_MAGIC.every((b, i) => bytes[i] === b)) return false;
  return bytes[4] === 2 && bytes[5] === 1;
}

export function parseElfHeader(bytes: Uint8Array): ElfHeader {
  if (bytes.length < EHDR_SIZE) {
    throw new ElfFormatError(`truncated ELF header: ${bytes.length} of ${EHDR_SIZE} bytes`);
  }
  if (!ELF_MAGIC.every((b, i) => bytes[i] === b)) {
    throw new ElfFormatError("not an ELF file: bad magic");
  }
  if (bytes[4] !== 2) throw new ElfFormatError("only ELF64 is supported (32-bit binaries are phase 2)");
  if (bytes[5] !== 1) throw new ElfFormatError("only little-endian ELF is supported");

  const dv = view(bytes);
  const type = dv.getUint16(16, true);
  if (type !== 2 && type !== 3) {
    throw new ElfFormatError(`unsupported ELF type ${type}: expected ET_EXEC (2) or ET_DYN (3)`);
  }
  const machine = dv.getUint16(18, true);
  if (machine !== 0x3e && machine !== 0xb7) {
    throw new ElfFormatError(`unsupported machine 0x${machine.toString(16)}: expected x86-64 (0x3e) or aarch64 (0xb7)`);
  }

  return {
    elfClass: 2,
    type,
    machine,
    entry: dv.getBigUint64(24, true),
    programHeaderOffset: dv.getBigUint64(32, true),
    programHeaderSize: dv.getUint16(54, true),
    programHeaderCount: dv.getUint16(56, true),
    sectionHeaderOffset: dv.getBigUint64(40, true),
    sectionHeaderSize: dv.getUint16(58, true),
    sectionHeaderCount: dv.getUint16(60, true),
    sectionNameIndex: dv.getUint16(62, true),
  };
}

export function parseProgramHeaders(bytes: Uint8Array, header: ElfHeader): ProgramHeader[] {
  if (header.programHeaderSize !== PHDR_SIZE) {
    throw new ElfFormatError(`unexpected program header size ${header.programHeaderSize}: expected ${PHDR_SIZE}`);
  }
  const start = Number(header.programHeaderOffset);
  const total = start + header.programHeaderCount * PHDR_SIZE;
  if (total > bytes.length) {
    throw new ElfFormatError("program header table runs past the end of the file");
  }
  const dv = view(bytes);
  const headers: ProgramHeader[] = [];
  for (let i = 0; i < header.programHeaderCount; i += 1) {
    const base = start + i * PHDR_SIZE;
    headers.push({
      type: dv.getUint32(base, true),
      flags: dv.getUint32(base + 4, true),
      offset: dv.getBigUint64(base + 8, true),
      virtualAddress: dv.getBigUint64(base + 16, true),
      fileSize: dv.getBigUint64(base + 32, true),
      memorySize: dv.getBigUint64(base + 40, true),
      alignment: dv.getBigUint64(base + 48, true),
    });
  }
  return headers;
}

/** The segments a translator must map: PT_LOAD only, in file order. */
export function loadableSegments(headers: readonly ProgramHeader[]): ProgramHeader[] {
  return headers.filter((h) => h.type === PT_LOAD);
}

/** Read a NUL-terminated string out of a byte table, refusing a read that runs off the end. */
function stringAt(bytes: Uint8Array, start: number, end: number): string {
  if (start < 0 || start >= end || end > bytes.length) {
    throw new ElfFormatError(`string offset ${start} is outside its table (${end} bytes)`);
  }
  let stop = start;
  while (stop < end && bytes[stop] !== 0) stop += 1;
  if (stop === end) {
    throw new ElfFormatError(`string at offset ${start} is not NUL-terminated inside its table`);
  }
  let name = "";
  for (let i = start; i < stop; i += 1) name += String.fromCharCode(bytes[i] as number);
  return name;
}

/**
 * Parse the section header table and resolve every section name.
 *
 * Names live in a string table that is itself a section (`e_shstrndx`), which is why this resolves in two passes:
 * read the raw entries, then read the name table they point into.
 */
export function parseSectionHeaders(bytes: Uint8Array, header: ElfHeader): SectionHeader[] {
  if (header.sectionHeaderSize !== SHDR_SIZE) {
    throw new ElfFormatError(`unexpected section header size ${header.sectionHeaderSize}: expected ${SHDR_SIZE}`);
  }
  if (header.sectionHeaderCount === 0) {
    throw new ElfFormatError("no section headers: stripped with --strip-all, or extended numbering (unsupported)");
  }
  if (header.sectionNameIndex >= header.sectionHeaderCount) {
    throw new ElfFormatError(
      `e_shstrndx ${header.sectionNameIndex} is past the ${header.sectionHeaderCount} sections`,
    );
  }
  const start = Number(header.sectionHeaderOffset);
  const total = start + header.sectionHeaderCount * SHDR_SIZE;
  if (total > bytes.length) {
    throw new ElfFormatError("section header table runs past the end of the file");
  }

  const dv = view(bytes);
  const raw = [];
  for (let i = 0; i < header.sectionHeaderCount; i += 1) {
    const base = start + i * SHDR_SIZE;
    raw.push({
      nameOffset: dv.getUint32(base, true),
      type: dv.getUint32(base + 4, true),
      flags: dv.getBigUint64(base + 8, true),
      address: dv.getBigUint64(base + 16, true),
      offset: dv.getBigUint64(base + 24, true),
      size: dv.getBigUint64(base + 32, true),
      link: dv.getUint32(base + 40, true),
      entrySize: dv.getBigUint64(base + 56, true),
    });
  }

  const names = raw[header.sectionNameIndex];
  if (names === undefined) {
    throw new ElfFormatError(`e_shstrndx ${header.sectionNameIndex} points at no section`);
  }
  if (names.type !== SHT_STRTAB) {
    throw new ElfFormatError(`section name table is type ${names.type}, not SHT_STRTAB (3)`);
  }
  const nameStart = Number(names.offset);
  const nameEnd = nameStart + Number(names.size);

  return raw.map((entry, index) => ({
    index,
    name: entry.nameOffset === 0 ? "" : stringAt(bytes, nameStart + entry.nameOffset, nameEnd),
    type: entry.type,
    flags: entry.flags,
    address: entry.address,
    offset: entry.offset,
    size: entry.size,
    link: entry.link,
    entrySize: entry.entrySize,
  }));
}

/** The first section with this name, or undefined. */
export function findSection(sections: readonly SectionHeader[], name: string): SectionHeader | undefined {
  return sections.find((section) => section.name === name);
}

/**
 * Read a symbol table (`.symtab`, or `.dynsym` with `dynamic: true`).
 *
 * A stripped binary has no symbols, and that is a refusal rather than an empty list: "no symbols" and "symbols I
 * failed to read" must not look the same to the caller.
 */
export function parseSymbols(
  bytes: Uint8Array,
  sections: readonly SectionHeader[],
  options: { dynamic?: boolean } = {},
): ElfSymbol[] {
  const wanted = options.dynamic ? SHT_DYNSYM : SHT_SYMTAB;
  const label = options.dynamic ? ".dynsym" : ".symtab";
  const table = sections.find((section) => section.type === wanted);
  if (table === undefined) {
    throw new ElfFormatError(`no ${label} section: this object carries no ${options.dynamic ? "dynamic" : "static"} symbols`);
  }
  if (table.entrySize !== BigInt(SYM_SIZE)) {
    throw new ElfFormatError(`${label} entry size ${table.entrySize} is not ${SYM_SIZE}: not an Elf64_Sym table`);
  }
  const strings = sections[table.link];
  if (strings === undefined) {
    throw new ElfFormatError(`${label} links to section ${table.link}, which does not exist`);
  }
  if (strings.type !== SHT_STRTAB) {
    throw new ElfFormatError(`${label} links to section ${table.link} (${strings.name}), not a string table`);
  }
  const tableStart = Number(table.offset);
  const tableSize = Number(table.size);
  if (tableStart + tableSize > bytes.length) {
    throw new ElfFormatError(`${label} runs past the end of the file`);
  }
  const stringStart = Number(strings.offset);
  const stringEnd = stringStart + Number(strings.size);

  const dv = view(bytes);
  const symbols: ElfSymbol[] = [];
  const count = tableSize / SYM_SIZE;
  for (let i = 1; i < count; i += 1) {
    // index 0 is the reserved null symbol, by spec
    const base = tableStart + i * SYM_SIZE;
    const info = dv.getUint8(base + 4);
    symbols.push({
      name: stringAt(bytes, stringStart + dv.getUint32(base, true), stringEnd),
      value: dv.getBigUint64(base + 8, true),
      size: dv.getBigUint64(base + 16, true),
      bind: info >> 4,
      type: info & 0x0f,
      sectionIndex: dv.getUint16(base + 6, true),
    });
  }
  return symbols;
}

/**
 * Relocations, which is the last thing a linker has to understand before a loader can exist.
 *
 * Only `RELA` is supported, which is what x86-64 and aarch64 both use. `REL` (the 32-bit Intel style, where the addend
 * lives at the relocation site in memory rather than in the file) is refused by name: reading it as if it were `RELA`
 * would produce relocations with addend zero, which link to the wrong place instead of failing.
 */
export interface Relocation {
  readonly offset: bigint;
  readonly symbolIndex: number;
  readonly type: number;
  /** The symbol's name, when the table links to one this file can read. */
  readonly symbolName: string;
  readonly addend: bigint;
}

const SHT_RELA = 4;
const SHT_REL = 9;
const RELA_SIZE = 24;

export function parseRelocations(
  bytes: Uint8Array,
  sections: readonly SectionHeader[],
  options: { dynamic?: boolean } = {},
): Relocation[] {
  const label = options.dynamic ? "dynamic relocations" : "relocations";
  const sites = sections.filter(
    (section) => section.type === SHT_RELA && (!options.dynamic || section.name.includes("dyn")),
  );
  if (sites.length === 0) {
    const relStyle = sections.filter((section) => section.type === SHT_REL);
    if (relStyle.length > 0) {
      throw new ElfFormatError(
        `only RELA relocations are supported: ${relStyle.map((s) => s.name).join(", ")} uses REL, ` +
          "whose addend lives in memory at the relocation site and cannot be read from the file",
      );
    }
    throw new ElfFormatError(`no ${label} in this object`);
  }

  const symbolsBySection = new Map<number, ElfSymbol[]>();
  const out: Relocation[] = [];
  for (const site of sites) {
    if (Number(site.entrySize) !== RELA_SIZE) {
      throw new ElfFormatError(
        `${site.name} has ${site.entrySize.toString()}-byte entries: expected ${RELA_SIZE} for Elf64_Rela`,
      );
    }
    const symbolSection = sections[site.link];
    if (symbolSection === undefined) {
      throw new ElfFormatError(`${site.name} links to section ${site.link}, which does not exist`);
    }
    if (symbolSection.type !== SHT_SYMTAB && symbolSection.type !== SHT_DYNSYM) {
      throw new ElfFormatError(`${site.name} links to ${symbolSection.name}, which is not a symbol table`);
    }
    let symbols = symbolsBySection.get(site.link);
    if (symbols === undefined) {
      symbols = parseSymbols(bytes, sections, { dynamic: symbolSection.type === SHT_DYNSYM });
      symbolsBySection.set(site.link, symbols);
    }

    const start = Number(site.offset);
    const size = Number(site.size);
    if (start + size > bytes.length) {
      throw new ElfFormatError(`${site.name} runs past the end of the file`);
    }
    const dv = view(bytes);
    for (let offset = start; offset < start + size; offset += RELA_SIZE) {
      const info = dv.getBigUint64(offset + 8, true);
      const symbolIndex = Number(info >> 32n);
      // symbol 0 is the reserved null symbol, so index 0 means "no symbol" rather than "symbols[0]"
      const symbol = symbolIndex === 0 ? undefined : symbols[symbolIndex - 1];
      out.push({
        offset: dv.getBigUint64(offset, true),
        symbolIndex,
        type: Number(info & 0xffffffffn),
        symbolName: symbol?.name ?? "",
        addend: dv.getBigInt64(offset + 16, true),
      });
    }
  }
  return out;
}
