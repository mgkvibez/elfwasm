/**
 * Placing `PT_LOAD` segments into WASM memory, which is the last reading step before a translator can run.
 *
 * A static binary assumes its segments sit at the virtual addresses named in the program headers, and that everything
 * between them is either mapped with the same bytes or genuinely absent. WASM memory is a single flat, byte-addressed
 * buffer with no per-page protection and no notion of an unmapped hole, so the translation happens here once, in one
 * place, instead of being rediscovered by every later stage.
 *
 * Two things are deliberately explicit:
 *
 * 1. The image's base is the lowest segment address rounded down to a WASM page, so every segment keeps its exact
 *    relative distance from every other. A translator can therefore add a constant offset and be right.
 * 2. Permissions are carried, not enforced. WASM cannot deny a write to a page, so `write`/`execute` here describe what
 *    the binary asked for. A translator that relies on the hardware to enforce it is wrong, and the type says so.
 */

import { ElfFormatError, loadableSegments, parseElfHeader, parseProgramHeaders } from "./elf.js";
import type { ElfHeader, ProgramHeader } from "./elf.js";

/** WASM memory is measured in 64 KiB pages, and that rounding is the only place the segment layout is disturbed. */
export const WASM_PAGE_SIZE = 64 * 1024;

/** WASM's linear memory is 32-bit, so no virtual address at or above this can ever be represented. */
const WASM_ADDRESS_LIMIT = 2 ** 32;

const PF_X = 1;
const PF_W = 2;
const PF_R = 4;

export interface SegmentPermissions {
  readonly read: boolean;
  readonly write: boolean;
  readonly execute: boolean;
}

/** One segment as it sits inside the WASM buffer, with both address spaces recorded. */
export interface MappedSegment {
  readonly virtualAddress: bigint;
  readonly fileOffset: bigint;
  /** Byte offset inside the WASM memory: `virtualAddress - baseAddress`. */
  readonly wasmAddress: number;
  readonly fileSize: bigint;
  readonly memorySize: bigint;
  readonly permissions: SegmentPermissions;
}

/**
 * The slice of `WebAssembly.Memory` this module actually uses.
 *
 * The project compiles with `types: []` and no DOM lib, deliberately: a translator that only works in a browser is a
 * translator that cannot be tested here. Declaring the shape keeps the dependency at zero and the surface honest.
 */
export interface LinearMemory {
  readonly buffer: ArrayBuffer;
}

declare const WebAssembly: {
  readonly Memory: new (descriptor: { initial: number }) => LinearMemory;
};

export interface MemoryImage {
  /** The virtual address that sits at WASM offset 0. Always a multiple of {@link WASM_PAGE_SIZE}. */
  readonly baseAddress: bigint;
  /** Total bytes to reserve, rounded up to a whole number of WASM pages. */
  readonly wasmBytes: number;
  readonly pages: number;
  /** The ELF entry point, translated into a WASM offset. */
  readonly entryAddress: number;
  readonly segments: readonly MappedSegment[];
}

function permissionsOf(flags: number): SegmentPermissions {
  return {
    read: (flags & PF_R) !== 0,
    // PF_X and PF_W are mutually exclusive on real hardware; the flag word is what the file says either way
    write: (flags & PF_W) !== 0,
    execute: (flags & PF_X) !== 0,
  };
}

function hex(value: bigint): string {
  return `0x${value.toString(16)}`;
}

/**
 * Build the image description without allocating anything.
 *
 * Every refusal names the segment it is about, because "bad file" tells a caller nothing when a linker is the thing
 * that produced it.
 */
export function buildMemoryImage(
  bytes: Uint8Array,
  header: ElfHeader,
  segments: readonly ProgramHeader[],
): MemoryImage {
  const loadable = loadableSegments(segments);
  if (loadable.length === 0) {
    throw new ElfFormatError("no PT_LOAD segments: there is nothing to map into memory");
  }

  let lowest = loadable[0]!.virtualAddress;
  let highest = 0n;
  for (const [index, segment] of loadable.entries()) {
    if (segment.fileSize > segment.memorySize) {
      throw new ElfFormatError(
        `segment ${index} at ${hex(segment.virtualAddress)} has p_filesz ${segment.fileSize} larger than ` +
          `p_memsz ${segment.memorySize}: the file cannot supply more than the image reserves`,
      );
    }
    if (segment.offset + segment.fileSize > BigInt(bytes.length)) {
      throw new ElfFormatError(
        `segment ${index} at ${hex(segment.virtualAddress)} runs past the end of the file: ` +
          `${segment.offset + segment.fileSize} of ${bytes.length} bytes`,
      );
    }
    if (segment.alignment > 1n) {
      // ELF requires this, and a segment that breaks it would land at a different page offset than the linker assumed
      if ((segment.virtualAddress - segment.offset) % segment.alignment !== 0n) {
        throw new ElfFormatError(
          `segment ${index} at ${hex(segment.virtualAddress)} is not congruent to its file offset ` +
            `${segment.offset} modulo p_align ${segment.alignment}`,
        );
      }
    }
    if (segment.virtualAddress < lowest) lowest = segment.virtualAddress;
    const end = segment.virtualAddress + segment.memorySize;
    if (end > highest) highest = end;
  }

  const page = BigInt(WASM_PAGE_SIZE);
  const baseAddress = (lowest / page) * page;
  if (highest - baseAddress > BigInt(WASM_ADDRESS_LIMIT)) {
    throw new ElfFormatError(
      `image would need ${highest - baseAddress} bytes of address space, past WASM's 32-bit limit of ${WASM_ADDRESS_LIMIT}`,
    );
  }

  const ordered = [...loadable].sort((a, b) => (a.virtualAddress < b.virtualAddress ? -1 : 1));
  for (let i = 1; i < ordered.length; i += 1) {
    const previous = ordered[i - 1]!;
    const current = ordered[i]!;
    const previousEnd = previous.virtualAddress + previous.memorySize;
    if (current.virtualAddress < previousEnd) {
      throw new ElfFormatError(
        `segments overlap in memory: ${hex(previous.virtualAddress)}..${hex(previousEnd)} and ` +
          `${hex(current.virtualAddress)}..${hex(current.virtualAddress + current.memorySize)}`,
      );
    }
  }

  const wasmBytes = Math.ceil((Number(highest - baseAddress) || 1) / WASM_PAGE_SIZE) * WASM_PAGE_SIZE;

  const mapped: MappedSegment[] = ordered.map((segment) => ({
    virtualAddress: segment.virtualAddress,
    fileOffset: segment.offset,
    wasmAddress: Number(segment.virtualAddress - baseAddress),
    fileSize: segment.fileSize,
    memorySize: segment.memorySize,
    permissions: permissionsOf(segment.flags),
  }));

  const entrySegment = mapped.find(
    (segment) =>
      segment.permissions.execute &&
      header.entry >= segment.virtualAddress &&
      header.entry < segment.virtualAddress + segment.memorySize,
  );
  if (entrySegment === undefined) {
    throw new ElfFormatError(
      `entry point ${hex(header.entry)} is not inside any executable PT_LOAD segment, so there is nothing to start`,
    );
  }

  return {
    baseAddress,
    wasmBytes,
    pages: wasmBytes / WASM_PAGE_SIZE,
    entryAddress: Number(header.entry - baseAddress),
    segments: mapped,
  };
}

/** Parse an ELF buffer and produce its image description in one step, for callers that only have bytes. */
export function imageOf(bytes: Uint8Array): MemoryImage {
  const header = parseElfHeader(bytes);
  return buildMemoryImage(bytes, header, parseProgramHeaders(bytes, header));
}

/** Reserve the WASM memory the image needs. Uninitialised pages read as zero, which is what a BSS wants. */
export function createMemory(image: MemoryImage): LinearMemory {
  return new WebAssembly.Memory({ initial: image.pages });
}

/**
 * Copy the file's segment bytes into an existing memory, and zero the tail of every segment whose `p_memsz` exceeds its
 * `p_filesz`.
 *
 * The explicit zero-fill is not redundant. A fresh `WebAssembly.Memory` happens to start zeroed, so a loader that only
 * copies bytes passes its first test and corrupts a reused buffer; BSS holding a previous process's values is exactly
 * the kind of bug that survives to the field.
 */
export function loadImage(bytes: Uint8Array, image: MemoryImage, memory: LinearMemory): void {
  const buffer = new Uint8Array(memory.buffer);
  // growing a memory detaches the old buffer, so the requirement is checked before any write rather than mid-copy
  if (buffer.length < image.wasmBytes) {
    throw new ElfFormatError(
      `WASM memory is ${buffer.length} bytes but the image needs ${image.wasmBytes}: refusing to write a partial image`,
    );
  }
  for (const segment of image.segments) {
    const start = segment.wasmAddress;
    const fileEnd = start + Number(segment.fileSize);
    const memoryEnd = start + Number(segment.memorySize);
    buffer.set(bytes.subarray(Number(segment.fileOffset), Number(segment.fileOffset + segment.fileSize)), start);
    buffer.fill(0, fileEnd, memoryEnd);
  }
}

/**
 * Translate a virtual address to a WASM offset.
 *
 * An address outside every mapped segment is refused rather than translated, because a flat buffer would happily return
 * whatever happened to be at that offset: a wild read that returns plausible bytes is worse than a crash.
 */
export function toWasmAddress(image: MemoryImage, virtualAddress: bigint): number {
  for (const segment of image.segments) {
    if (virtualAddress >= segment.virtualAddress && virtualAddress < segment.virtualAddress + segment.memorySize) {
      return segment.wasmAddress + Number(virtualAddress - segment.virtualAddress);
    }
  }
  throw new ElfFormatError(
    `virtual address ${hex(virtualAddress)} is not mapped by any PT_LOAD segment: ` +
      `mapped ranges are ${image.segments.map((s) => `${hex(s.virtualAddress)}..${hex(s.virtualAddress + s.memorySize)}`).join(", ")}`,
  );
}

/**
 * The image as an address space a translator can ask questions of.
 *
 * `toWasmAddress` throws, which is right for a loader and wrong for a translator: the translator is deciding whether an
 * operand is translatable, so it needs the answer as a value. The refusal is not lost, it moves to the caller that has
 * the instruction's offset and can name it.
 */
export interface AddressSpace {
  /** The virtual address that sits at WASM offset 0, which a run-time address has to subtract. */
  readonly baseAddress: bigint;
  /** Pages the image itself occupies, which is where a stack placed above it may not grow down into. */
  readonly pages: number;
  /** WASM byte offset of this virtual address, or undefined when no segment maps it. */
  offsetOf(virtualAddress: bigint): number | undefined;
}

export function addressSpaceOf(image: MemoryImage): AddressSpace {
  return {
    baseAddress: image.baseAddress,
    pages: image.pages,
    offsetOf(virtualAddress: bigint): number | undefined {
      for (const segment of image.segments) {
        if (virtualAddress >= segment.virtualAddress && virtualAddress < segment.virtualAddress + segment.memorySize) {
          return segment.wasmAddress + Number(virtualAddress - segment.virtualAddress);
        }
      }
      return undefined;
    },
  };
}
