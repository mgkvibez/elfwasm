/**
 * The loader: place the image, copy the segments in, and apply the relocations a position-independent file needs.
 *
 * A non-PIE executable is fully resolved by its linker: every address in it is the address it will run at, so copying the
 * segments into memory is the whole job. A position-independent executable is not: its code and data hold *offsets*, and a
 * table of relocations says which eight-byte slots have to be turned into addresses once the load address is known. Skip
 * that and the program runs with zeroes where its pointers should be, which looks like a working program reading a null
 * array rather than like a broken load.
 *
 * The subset here is `R_X86_64_RELATIVE` and the answer is a computed address, not a resolved symbol. Everything else is
 * refused by name, because a loader that guessed at a `JUMP_SLOT` would silently point a call at whatever was at the slot.
 */

import {
  ElfFormatError,
  SHT_REL,
  SHT_RELA,
  parseElfHeader,
  parseRelocations,
  parseSectionHeaders,
} from "./elf.js";
import type { Relocation } from "./elf.js";
import { createMemory, imageOf, loadImage } from "./image.js";
import type { LinearMemory, MemoryImage, MappedSegment } from "./image.js";

/**
 * Where a position-independent image lands by default: the lowest address a Linux program is conventionally linked at.
 *
 * Not zero, and that is the point. A PIE is linked at zero, so loading it where it says it wants to be would map the null
 * page: a program that dereferences a null pointer would read the ELF header and keep going instead of trapping, and every
 * "is this pointer valid" check in the translated code would answer yes.
 */
export const DEFAULT_LOAD_BASE = 0x400000n;

const R_X86_64_NONE = 0;
const R_X86_64_RELATIVE = 8;

/** The types this loader gives a name to, so a refusal says which relocation it is rather than which number. */
const RELOCATION_NAMES: Readonly<Record<number, string>> = {
  0: "R_X86_64_NONE",
  1: "R_X86_64_64",
  2: "R_X86_64_PC32",
  5: "R_X86_64_COPY",
  6: "R_X86_64_GLOB_DAT",
  7: "R_X86_64_JUMP_SLOT",
  8: "R_X86_64_RELATIVE",
  9: "R_X86_64_GOTPCREL",
  10: "R_X86_64_32",
  11: "R_X86_64_32S",
  37: "R_X86_64_IRELATIVE",
};

/** Why a type outside the subset cannot be applied, where there is something specific and true to say. */
const UNSUPPORTED_REASONS: Readonly<Record<number, string>> = {
  1: "it names a symbol's address, and this loader resolves no symbols",
  5: "it copies a definition out of another object, which only a dynamic linker has",
  6: "it names a symbol a dynamic linker would resolve, and a statically loaded image has no resolver",
  7: "it names a symbol a dynamic linker would resolve through a lazy binding, and there is no resolver here",
  37:
    "it names a function the loader itself would have to call before the program starts, and this loader runs no code " +
    "while loading",
};

function hex(value: bigint): string {
  return `0x${value.toString(16)}`;
}

function nameOf(type: number): string {
  return RELOCATION_NAMES[type] ?? `relocation type ${type}`;
}

/** The segment a slot sits in, refused unless all `width` bytes are inside the segment the image reserved for it. */
function segmentFor(image: MemoryImage, address: bigint, width: number): MappedSegment {
  for (const segment of image.segments) {
    const start = segment.virtualAddress;
    const end = start + segment.memorySize;
    if (address < start || address >= end) continue;
    if (address + BigInt(width) > end) {
      throw new ElfFormatError(
        `a relocation at ${hex(address)} writes ${width} bytes but the segment at ${hex(start)} ends at ${hex(end)}: ` +
          "the slot runs off the end of the memory the image reserved",
      );
    }
    return segment;
  }
  throw new ElfFormatError(
    `a relocation at ${hex(address)} is not inside any mapped segment, so there is nothing to write: mapped ranges are ` +
      image.segments
        .map((segment) => `${hex(segment.virtualAddress)}..${hex(segment.virtualAddress + segment.memorySize)}`)
        .join(", "),
  );
}

export interface ApplyOptions {
  readonly image: MemoryImage;
  readonly memory: LinearMemory;
}

/**
 * Write the relocations into the loaded image, and say how many were applied.
 *
 * A relative relocation is `slot = loadBias + addend`: the addend is the target's address in the file's own layout, so once
 * the whole image moves by the bias, the pointer has to move by the same amount. Both halves of the entry are in the file's
 * address space, not the one the image was loaded in: `r_offset` sits at `loadBias + r_offset` and the value written is
 * `loadBias + r_addend`. Reading the offset as a load-time address is the mistake that makes every relocation land 0x400000
 * bytes below where it belongs, which is inside some other segment often enough to look like a wrong value rather than a
 * missed rebase.
 *
 * A slot in a segment that is not writable is refused. The file asked for those pages to be read-only, and a loader that
 * ignored the flag would produce a program whose constants had been quietly overwritten by its own relocations. A real ELF
 * puts these slots in `RELRO` pages, which the program header marks writable precisely because the loader writes them before
 * the program runs, so the flag is the check that the mapping is the one the linker had in mind.
 */
export function applyRelocations(relocations: readonly Relocation[], options: ApplyOptions): number {
  const { image, memory } = options;
  const buffer = new Uint8Array(memory.buffer);
  const view = new DataView(memory.buffer);
  let applied = 0;

  for (const relocation of relocations) {
    if (relocation.type === R_X86_64_NONE) continue;

    // the offset is in the file's own address space, like the addend, so both move with the image
    const target = image.loadBias + relocation.offset;
    const at = hex(target);
    if (relocation.type !== R_X86_64_RELATIVE) {
      const reason = UNSUPPORTED_REASONS[relocation.type];
      throw new ElfFormatError(
        `${nameOf(relocation.type)} at ${at} is not applied` +
          (reason === undefined ? ": this loader implements R_X86_64_RELATIVE only" : `: ${reason}`),
      );
    }

    const segment = segmentFor(image, target, 8);
    if (!segment.permissions.write) {
      throw new ElfFormatError(
        `a relocation at ${at} writes into the segment at ${hex(segment.virtualAddress)}, which the file marks ` +
          "read-only: the slot belongs in a writable segment, so either the mapping or the file is not what it claims",
      );
    }

    const offset = segment.wasmAddress + Number(target - segment.virtualAddress);
    if (offset < 0 || offset + 8 > buffer.length) {
      throw new ElfFormatError(
        `a relocation at ${at} maps to buffer offset ${offset}, past the ${buffer.length} bytes of memory`,
      );
    }
    view.setBigUint64(offset, image.loadBias + relocation.addend, true);
    applied += 1;
  }

  return applied;
}

export interface LoadOptions {
  /** Where to put virtual address zero. Defaults to {@link DEFAULT_LOAD_BASE}. */
  readonly base?: bigint;
}

export interface LoadedExecutable {
  readonly image: MemoryImage;
  readonly memory: LinearMemory;
  /** How many relocations were written, which is zero for a non-PIE executable. */
  readonly relocations: number;
}

/**
 * Find the relocations a file carries, through its section headers.
 *
 * A stripped file has none to read, and a position-independent one is then refused rather than loaded: its pointers would
 * stay at their link-time offsets, which is not a program that runs slightly wrong but one whose every indirect reference is
 * zero. A non-PIE file legitimately has none at all, and that is not an error.
 */
function relocationsOf(bytes: Uint8Array): Relocation[] {
  const header = parseElfHeader(bytes);
  const sections = parseSectionHeaders(bytes, header);
  if (!sections.some((section) => section.type === SHT_RELA || section.type === SHT_REL)) {
    if (header.type === 3) {
      throw new ElfFormatError(
        "this is a position-independent executable with no relocation sections, so its pointers cannot be fixed up: " +
          "loading it would run a program whose every address is an offset from zero",
      );
    }
    return [];
  }
  return parseRelocations(bytes, sections);
}

/**
 * Load an executable into a fresh memory: image, bytes, relocations, in that order.
 *
 * The order is the load-bearing part. Relocations are written *after* the segments are copied, because they overwrite bytes
 * the file supplied; and they are written before anything executes, because a program that starts with a half-fixed-up image
 * is a program whose first indirect call goes wherever the zeroes point.
 */
export function loadExecutable(bytes: Uint8Array, options: LoadOptions = {}): LoadedExecutable {
  const base = options.base ?? DEFAULT_LOAD_BASE;
  if (base === 0n) {
    throw new ElfFormatError(
      "refusing to load at virtual address 0: the null page would be mapped, so a translated program that dereferences a " +
        "null pointer would read the image instead of trapping",
    );
  }

  const image = imageOf(bytes, { base });
  const memory = createMemory(image);
  loadImage(bytes, image, memory);
  const relocations = applyRelocations(relocationsOf(bytes), { image, memory });
  return { image, memory, relocations };
}
