/**
 * The syscall layer: Linux's own interface, answered by JavaScript.
 *
 * A translated program reaches the kernel through `syscall`, which becomes a call to an imported host function taking the call
 * number and six arguments. The host gets *virtual* addresses, exactly as the kernel does, and converts them with the image's
 * address space: a `write` whose buffer is outside the image is `-EFAULT` rather than a read of something that happens to be
 * there.
 *
 * Six calls are implemented, and that is the whole claim: `write`, `exit`, `exit_group`, `arch_prctl` for the thread pointer,
 * and `brk` and `mmap` for allocation. Everything else returns `-ENOSYS`, which is what a real kernel returns for a call it does
 * not have, so a program that needs one fails the way it would on a machine without it instead of silently reading zeroes.
 */

import { addressSpaceOf } from "./image.js";
import type { AddressSpace, LinearMemory, MemoryImage } from "./image.js";

/**
 * The slice of the WASM API a machine needs, declared for the same reason `image.ts` declares its own: this project compiles
 * with no DOM lib and no ambient types, so the dependency stays at zero and the surface stays honest.
 */
declare const WebAssembly: {
  readonly Memory: new (descriptor: { initial: number }) => LinearMemory;
  readonly Module: new (bytes: Uint8Array) => unknown;
  readonly Instance: new (
    module: unknown,
    imports?: { readonly env: Record<string, unknown> },
  ) => { readonly exports: Record<string, unknown> };
  /** A mutable `i64` global, which is how a translated module's thread pointer comes back to the host. */
  readonly Global: new (descriptor: { readonly value: "i64"; readonly mutable: boolean }, initial: bigint) => {
    value: bigint;
  };
};

/** The `WebAssembly.Global` a module exports for its thread pointer, as much of it as this layer writes. */
interface ThreadPointer {
  value: bigint;
}

/** The errno values this layer can return, in their negative form, which is how a syscall reports failure. */
export const ERRNO = {
  ENOMEM: -12n,
  EFAULT: -14n,
  EINVAL: -22n,
  ENOSYS: -38n,
  EBADF: -9n,
  EIO: -5n,
} as const;

/** Linux's x86-64 call numbers, named so the shim does not carry bare integers. */
export const SYSCALLS = {
  write: 1n,
  mmap: 9n,
  brk: 12n,
  archPrctl: 158n,
  exit: 60n,
  exitGroup: 231n,
} as const;

/**
 * The two `arch_prctl` operations that concern the thread pointer.
 *
 * `ARCH_SET_FS` points `%fs` at a block, and `ARCH_GET_FS` reports where it points, which is how a program that inherited a
 * thread pointer finds it again. The GDT-entry forms are the 32-bit interface and are not named here: on x86-64 they are not
 * how a thread pointer is set, and answering them would be answering a question this kernel cannot have been asked.
 */
export const ARCH_PRCTL = {
  setFs: 0x1002n,
  getFs: 0x1003n,
} as const;

/**
 * The `mmap` flags this layer reads.
 *
 * A private anonymous mapping is the one kind it can serve: there is no file behind the address space, so a mapping of a file
 * has nothing to map, and `MAP_FIXED` asks for an address that would land on the image.
 */
export const MMAP = {
  fixed: 0x10n,
  anonymous: 0x20n,
} as const;

/** Linux's page size, which is the granularity `mmap` and `munmap` work in, and is not a WASM page. */
export const PAGE_SIZE = 0x1000n;
/** A WASM page, which is what the memory is measured in. */
const WASM_PAGE = 0x10000n;

/** Thrown when the program exits, because an exit never returns: the kernel replaces the process, so this throws instead. */
export class Exit extends Error {
  readonly code: number;

  constructor(code: number) {
    super(`the program exited with status ${code}`);
    this.name = "Exit";
    this.code = code;
  }
}

export interface MachineOptions {
  /** The mapped image, so a virtual address can be turned into an offset into the imported memory. */
  readonly image: MemoryImage;
  /** Pages to allocate. Defaults to the image's own pages, so a program with no stack gets exactly the image's memory. */
  readonly pages?: number;
  /**
   * The memory to run in, instead of creating one.
   *
   * A loaded image already lives in the memory its loader built, and the relocations were written into that buffer: running
   * in a fresh one would address a copy taken before any of them were applied, which reads as a program whose pointers are
   * all zero.
   */
  readonly memory?: LinearMemory;
  /**
   * Pages at the top of the memory that hold the stack, which the heap may not grow into.
   *
   * It has to be said rather than guessed: `%rsp` starts at the top of memory, so a break that grew to the top would hand a
   * program the pages its own frame is on. Defaults to zero, which is right for a program that was translated without a stack.
   */
  readonly stackPages?: number;
}

/** What a finished program produced: its status and the bytes it wrote to each standard descriptor. */
export interface Outcome {
  readonly exitCode: number;
  readonly stdout: Uint8Array;
  readonly stderr: Uint8Array;
}

/**
 * A machine: the memory the program addresses, the kernel it calls, and what it produced.
 *
 * The memory is created here rather than by the module, because the kernel has to read it too: `write` copies bytes out of the
 * same buffer the program wrote them into.
 */
export class Machine {
  readonly memory: LinearMemory;
  readonly addresses: AddressSpace;
  readonly imports: { readonly env: Record<string, unknown> };
  readonly #stdout: number[] = [];
  readonly #stderr: number[] = [];
  #exitCode: number | undefined;
  /**
   * The module's thread pointer, once it is instantiated.
   *
   * It is the module's global rather than a field of this class, and that is deliberate: the translated code reads it, so it has
   * to live where `global.get` can reach it. This layer only keeps a handle to write it, which is what `arch_prctl` does.
   */
  #thread: ThreadPointer | undefined;
  /** The program break: where `brk` hands out memory from, growing upward, and the byte after what it has given away. */
  #break: bigint;
  /** Where the next `mmap` goes, growing downward from the top of the free region, which is the direction Linux grows them. */
  #mappings: bigint;
  /** One past the last byte the heap may use: the top of the memory, less the pages the stack is sitting in. */
  readonly #heapLimit: bigint;

  constructor(options: MachineOptions) {
    this.addresses = addressSpaceOf(options.image);
    this.memory = options.memory ?? new WebAssembly.Memory({ initial: options.pages ?? options.image.pages });
    /*
     * The heap starts where the image's own pages end, so a virtual address and a WASM offset stay one addition apart all the
     * way up: `%fs` needed the same thing, and a heap that began at 0 would need its own mapping table for no benefit.
     */
    this.#break = this.addresses.baseAddress + BigInt(this.addresses.pages) * WASM_PAGE;
    this.#heapLimit =
      this.addresses.baseAddress + BigInt(this.memory.buffer.byteLength) - BigInt(options.stackPages ?? 0) * WASM_PAGE;
    // mappings grow down from the top, so the first one starts at the limit the break may not pass
    this.#mappings = this.#heapLimit;
    this.imports = {
      env: {
        memory: this.memory,
        syscall: (
          number: bigint,
          a1: bigint,
          a2: bigint,
          a3: bigint,
          a4: bigint,
          a5: bigint,
          a6: bigint,
        ): bigint => this.#syscall(number, a1, a2, a3, a4, a5, a6),
      },
    };
  }

  /** Runs a translated module and reports what it produced. */
  run(module: Uint8Array): Outcome {
    const instance = new WebAssembly.Instance(new WebAssembly.Module(module), this.imports);
    const entry = Object.values(instance.exports).find((exported) => typeof exported === "function");
    if (typeof entry !== "function") {
      throw new Error("the module has no exported function to run");
    }
    // the thread pointer starts at zero, which is what a process with no `%fs` base has, and `arch_prctl` moves it
    const thread = instance.exports.fs;
    this.#thread =
      typeof thread === "object" && thread !== null && "value" in thread ? (thread as ThreadPointer) : undefined;
    try {
      entry();
    } catch (error) {
      // an exit is not an error, it is the program finishing the only way this layer knows how
      if (!(error instanceof Exit)) throw error;
      if (this.#exitCode === undefined) this.#exitCode = error.code;
    }
    return {
      exitCode: this.#exitCode ?? 0,
      stdout: Uint8Array.from(this.#stdout),
      stderr: Uint8Array.from(this.#stderr),
    };
  }

  #syscall(number: bigint, a1: bigint, a2: bigint, a3: bigint, a4: bigint, a5: bigint, _a6: bigint): bigint {
    if (number === SYSCALLS.write) return this.#write(Number(a1), a2, Number(a3));
    if (number === SYSCALLS.archPrctl) return this.#archPrctl(a1, a2);
    if (number === SYSCALLS.brk) return this.#brk(a1);
    if (number === SYSCALLS.mmap) return this.#mmap(a1, a2, a3, a4, a5);
    if (number === SYSCALLS.exit || number === SYSCALLS.exitGroup) {
      const code = Number(a1) & 0xff;
      this.#exitCode = code;
      throw new Exit(code);
    }
    return ERRNO.ENOSYS;
  }

  /**
   * `arch_prctl`: point `%fs` somewhere, or report where it points.
   *
   * A module that did not export a thread pointer cannot have been translated with `%fs` addressing, so answering `-EINVAL` is
   * the truthful thing rather than pretending the call worked against a base nothing reads.
   */
  #archPrctl(operation: bigint, address: bigint): bigint {
    const thread = this.#thread;
    if (thread === undefined) return ERRNO.EINVAL;
    if (operation === ARCH_PRCTL.setFs) {
      thread.value = address;
      return 0n;
    }
    if (operation === ARCH_PRCTL.getFs) {
      const slot = this.bytesAt(address, 8);
      // a program asking where its thread pointer is has to supply somewhere to put the answer, and it may not be mapped
      if (slot === undefined) return ERRNO.EFAULT;
      new DataView(slot.buffer, slot.byteOffset, 8).setBigUint64(0, thread.value, true);
      return 0n;
    }
    return ERRNO.EINVAL;
  }

  #write(descriptor: number, address: bigint, count: number): bigint {
    if (descriptor !== 1 && descriptor !== 2) return ERRNO.EBADF;
    if (count < 0) return ERRNO.EINVAL;
    const bytes = this.bytesAt(address, count);
    if (bytes === undefined) return ERRNO.EFAULT;
    (descriptor === 1 ? this.#stdout : this.#stderr).push(...bytes);
    return BigInt(count);
  }

  /** Where the next byte `brk` hands out is, which a program can also read with `brk(0)`. */
  get breakAddress(): bigint {
    return this.#break;
  }

  /** The first page after the heap, which is either the top of the memory or the bottom of the stack. */
  get heapLimit(): bigint {
    return this.#heapLimit;
  }

  /**
   * `brk`: move the program break, and answer with where it is now.
   *
   * Linux answers with the break's *new* value on success and its *old* one on failure, which is how a caller tells them apart:
   * a failed `brk` returns what the previous `brk(0)` said. Shrinking is allowed, and so is a `brk` to exactly where it already
   * is. Growing into the stack or past the memory is not, and neither is a break below the image.
   */
  #brk(address: bigint): bigint {
    if (address === 0n) return this.#break;
    if (address < this.heapStart || address > this.#heapLimit || address > this.#mappings) return this.#break;
    if (address > this.#break) this.#zero(this.#break, address);
    this.#break = address;
    return this.#break;
  }

  /** The first byte after the image, which is where the heap begins. */
  get heapStart(): bigint {
    return this.addresses.baseAddress + BigInt(this.addresses.pages) * WASM_PAGE;
  }

  /**
   * `mmap`: a private anonymous mapping, growing down from the top of the free region.
   *
   * The address asked for is a hint and is ignored, which the interface allows. A file mapping has nothing to map here, and
   * `MAP_FIXED` names an address that may be the image, so both are refused rather than served as anonymous memory: a program
   * that maps a file and finds zeroes reads a working address containing the wrong bytes, which is worse than a failure.
   */
  #mmap(
    _address: bigint,
    length: bigint,
    _protection: bigint,
    flags: bigint,
    _descriptor: bigint,
  ): bigint {
    if ((flags & MMAP.fixed) !== 0n) return ERRNO.ENOSYS;
    if ((flags & MMAP.anonymous) === 0n) return ERRNO.ENOSYS;
    if (length === 0n) return ERRNO.EINVAL;
    // the address is a hint the kernel may ignore, the protection is a page bit WASM has not got, and an anonymous mapping
    // has no file behind it for the descriptor to name
    const rounded = (length + PAGE_SIZE - 1n) / PAGE_SIZE * PAGE_SIZE;
    const start = this.#mappings - rounded;
    if (start < this.#break) return ERRNO.ENOMEM;
    this.#mappings = start;
    this.#zero(start, start + rounded);
    return start;
  }

  /** Writes zeroes over a range, which is what a fresh page looks like: a new memory is zeroed, but a reused range is not. */
  #zero(from: bigint, to: bigint): void {
    const bytes = this.#linear(from, Number(to - from));
    bytes?.fill(0);
  }

  /**
   * The bytes at a virtual address, or `undefined` when the range is not entirely inside the memory.
   *
   * Inside the image the segments decide, because they are what a loader placed. Above it the addresses are the heap's, and
   * those are one addition apart from their WASM offsets all the way to the top.
   */
  bytesAt(address: bigint, count: number): Uint8Array | undefined {
    if (count === 0) return new Uint8Array(0);
    const inside = this.addresses.offsetOf(address);
    if (inside !== undefined) {
      return inside + count > this.memory.buffer.byteLength
        ? undefined
        : new Uint8Array(this.memory.buffer, inside, count);
    }
    return this.#linear(address, count);
  }

  #linear(address: bigint, count: number): Uint8Array | undefined {
    if (address < this.addresses.baseAddress) return undefined;
    const offset = Number(address - this.addresses.baseAddress);
    if (offset < 0 || offset + count > this.memory.buffer.byteLength) return undefined;
    return new Uint8Array(this.memory.buffer, offset, count);
  }
}
