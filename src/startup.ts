/**
 * The code that has to run before the program does.
 *
 * A `R_X86_64_IRELATIVE` relocation does not name a value, it names a *function*: the addend is the address of an ifunc
 * resolver, the loader calls it, and the address it returns is what goes in the slot. That is how glibc picks a `memcpy`
 * for the machine it finds itself on, and it is the last thing a static glibc PIE needs before its own `_start` can run.
 *
 * A resolver is ordinary x86-64 machine code sitting in the image, so running it means translating it and calling the
 * result, which is what this file does. Two things make that different from running a program:
 *
 * 1. There is no symbol table entry to give its length. What a resolver has is a `ret`, so the extent is decoded from the
 *    address until the first one: instruction by instruction, so a `0xc3` inside an immediate cannot cut it short.
 * 2. It is called, not run as an entry. The translator's module returns `%rax`, so `Machine.call` is calling the function.
 *
 * Anything the extent cannot cover is refused by name rather than run half-translated: a resolver longer than the limit, one
 * that never returns, or one whose `call` reaches outside its extent.
 */

import { DecodeError, decodeInstruction } from "./decode.js";
import type { Instruction } from "./decode.js";
import { ElfFormatError } from "./elf.js";
import { loadExecutable } from "./loader.js";
import type { LoadedExecutable, LoadOptions, Resolution } from "./loader.js";
import { Machine } from "./syscalls.js";
import type { LinearMemory, MemoryImage } from "./image.js";
import { translate } from "./translate.js";

/** How far to walk looking for a resolver's `ret`, in bytes. A resolver that is longer than this is not one. */
const DEFAULT_LIMIT = 512;

export interface ResolverOptions {
  /** The whole file's bytes, because the resolver's own code is in it. */
  readonly bytes: Uint8Array;
  readonly image: MemoryImage;
  readonly memory: LinearMemory;
  /** Pages reserved for the stack, when a resolver calls something that pushes. */
  readonly stackPages?: number;
  /** Answer `syscall` inside the resolver. Off by default: a resolver is startup code and asks the kernel nothing. */
  readonly syscalls?: boolean;
  /** Bytes to walk before giving up on finding a `ret`. */
  readonly limit?: number;
}

/** Where a virtual address sits in the file, and the segment it belongs to. */
function fileOffsetOf(image: MemoryImage, address: bigint): { index: number; end: number } {
  for (const segment of image.segments) {
    if (address < segment.virtualAddress || address >= segment.virtualAddress + segment.fileSize) continue;
    if (!segment.permissions.execute) {
      throw new ElfFormatError(
        `a resolver at ${hex(address)} is in a segment the file does not mark executable: a resolver is code, so either the ` +
          "address is not one or the mapping is not what the file says",
      );
    }
    const within = Number(address - segment.virtualAddress);
    const start = Number(segment.fileOffset) + within;
    return { index: start, end: Number(segment.fileOffset + segment.fileSize) };
  }
  throw new ElfFormatError(
    `a resolver at ${hex(address)} is not inside any file-backed segment, so its code cannot be read: the relocation names ` +
      "an address the file does not have bytes for",
  );
}

/**
 * A resolver's instructions, from its address to the `ret` that ends its answer.
 *
 * Control flow, not bytes. A linear scan from the start would stop at the first `ret` it *reads*, which is not the same as the
 * first one the resolver can *reach*: a resolver is allowed to end in a jump into another function, and a tail call like that
 * has a target the scan walks straight past. Decoding one instruction at a time and following the jumps instead means the body
 * is exactly the code reachable from the resolver's start, and a jump out of its own bytes lands on an instruction the
 * translation already has rather than between two of them.
 *
 * Offsets come back relative to the resolver's start, because that is what the translator measures a call or a jump against.
 */
function instructionsOf(
  bytes: Uint8Array,
  image: MemoryImage,
  address: bigint,
  limit: number,
): readonly Instruction[] {
  const { index, end } = fileOffsetOf(image, address);
  const bound = Math.min(end, index + limit);
  const decoded = new Map<number, Instruction>();
  const work: number[] = [index];
  let returned = false;
  while (work.length > 0) {
    const cursor = work.pop()!;
    if (decoded.has(cursor)) continue;
    if (cursor >= bound) {
      throw new ElfFormatError(
        `a resolver at ${hex(address)} has no ret in its first ${limit} bytes, so where it ends is not knowable: without an ` +
          "extent the translator would either stop inside it or run on into whatever follows",
      );
    }
    let instruction: Instruction;
    try {
      instruction = decodeInstruction(bytes, cursor);
    } catch (error) {
      if (!(error instanceof DecodeError)) throw error;
      throw new ElfFormatError(
        `a resolver at ${hex(address)} does not decode: ${error.message}. A resolver is translated before it runs, so an ` +
          "instruction outside the subset stops it here rather than halfway through its own work",
      );
    }
    decoded.set(cursor, instruction);
    if (instruction.op === "ret") {
      returned = true;
      continue;
    }
    // every instruction after a jump has a successor except the jump itself, which has only its target
    const displacement = instruction.operands[0];
    if (instruction.op.startsWith("j") && displacement !== undefined && displacement.kind === "immediate") {
      work.push(cursor + instruction.length + Number(displacement.value));
    }
    if (instruction.op !== "jmp") work.push(cursor + instruction.length);
  }
  if (!returned) {
    throw new ElfFormatError(
      `a resolver at ${hex(address)} has no ret in its first ${limit} bytes, so where it ends is not knowable: without an ` +
        "extent the translator would either stop inside it or run on into whatever follows",
    );
  }
  return [...decoded.values()]
    .sort((left, right) => left.offset - right.offset)
    .map((instruction) => ({ ...instruction, offset: instruction.offset - index }));
}

/**
 * A resolver that runs translated code.
 *
 * Each address is translated once and then called as often as the relocation table asks, because a resolver that is called
 * twice is a resolver whose code has not changed.
 */
export function translatedResolver(
  options: ResolverOptions,
): (target: bigint) => bigint {
  const { bytes, image, memory } = options;
  const machine = new Machine(
    options.stackPages === undefined ? { image, memory } : { image, memory, stackPages: options.stackPages },
  );
  const memoryPages = Math.floor(memory.buffer.byteLength / 0x10000);
  const modules = new Map<bigint, Uint8Array>();

  return (target: bigint): bigint => {
    let module = modules.get(target);
    if (module === undefined) {
      module = translate(instructionsOf(bytes, image, target, options.limit ?? DEFAULT_LIMIT), "resolver", {
        addresses: machine.addresses,
        codeAddress: target,
        code: { bytes, virtualAddress: image.baseAddress },
        memoryPages,
        syscalls: options.syscalls ?? false,
        ...(options.stackPages === undefined ? {} : { stackPages: options.stackPages }),
      });
      modules.set(target, module);
    }
    return machine.call(module);
  };
}

/**
 * A resolution that runs the image's own code, built the first time a relocation asks for one.
 *
 * The image and the memory only exist once the load has begun, so the resolver is built from the context `applyRelocations`
 * hands over rather than up front.
 */
export function resolveWithTranslatedCode(options: {
  readonly bytes: Uint8Array;
  readonly stackPages?: number;
  readonly syscalls?: boolean;
  readonly limit?: number;
}): Resolution {
  let built: { readonly memory: LinearMemory; readonly resolve: (target: bigint) => bigint } | undefined;
  return (target, context) => {
    if (built === undefined || built.memory !== context.memory) {
      built = { memory: context.memory, resolve: translatedResolver({ ...options, ...context }) };
    }
    return built.resolve(target);
  };
}

/**
 * Load an executable and resolve its ifuncs, which is a whole load including the code that decides what its pointers mean.
 *
 * A plain `loadExecutable` leaves the `IRELATIVE` slots unfilled or refuses them; this is the same load with a translator on
 * hand to run each resolver at the address the image was placed at.
 */
export function loadResolved(
  bytes: Uint8Array,
  options: LoadOptions & { readonly stackPages?: number; readonly syscalls?: boolean; readonly limit?: number } = {},
): LoadedExecutable {
  return loadExecutable(bytes, {
    ...(options.base === undefined ? {} : { base: options.base }),
    resolve: resolveWithTranslatedCode({
      bytes,
      ...(options.stackPages === undefined ? {} : { stackPages: options.stackPages }),
      ...(options.syscalls === undefined ? {} : { syscalls: options.syscalls }),
      ...(options.limit === undefined ? {} : { limit: options.limit }),
    }),
  });
}

function hex(value: bigint): string {
  return `0x${value.toString(16)}`;
}
