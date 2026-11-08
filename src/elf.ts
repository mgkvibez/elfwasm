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
}

export interface ProgramHeader {
  readonly type: number;
  readonly flags: number;
  readonly offset: bigint;
  readonly virtualAddress: bigint;
  readonly fileSize: bigint;
  readonly memorySize: bigint;
}

const ELF_MAGIC = [0x7f, 0x45, 0x4c, 0x46] as const;
const EHDR_SIZE = 64;
const PHDR_SIZE = 56;
const PT_LOAD = 1;

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
    });
  }
  return headers;
}

/** The segments a translator must map: PT_LOAD only, in file order. */
export function loadableSegments(headers: readonly ProgramHeader[]): ProgramHeader[] {
  return headers.filter((h) => h.type === PT_LOAD);
}
