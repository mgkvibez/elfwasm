/**
 * The syscall layer: Linux's own interface, answered by JavaScript.
 *
 * A translated program reaches the kernel through `syscall`, which becomes a call to an imported host function taking the call
 * number and six arguments. The host gets *virtual* addresses, exactly as the kernel does, and converts them with the image's
 * address space: a `write` whose buffer is outside the image is `-EFAULT` rather than a read of something that happens to be
 * there.
 *
 * Three calls are implemented, and that is the whole claim: `write`, `exit` and `exit_group`. Everything else returns `-ENOSYS`,
 * which is what a real kernel returns for a call it does not have, so a program that needs one fails the way it would on a
 * machine without it instead of silently reading zeroes.
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
};

/** The errno values this layer can return, in their negative form, which is how a syscall reports failure. */
export const ERRNO = {
  EFAULT: -14n,
  EINVAL: -22n,
  ENOSYS: -38n,
  EBADF: -9n,
  EIO: -5n,
} as const;

/** Linux's x86-64 call numbers, named so the shim does not carry bare integers. */
export const SYSCALLS = {
  write: 1n,
  exit: 60n,
  exitGroup: 231n,
} as const;

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

  constructor(options: MachineOptions) {
    this.addresses = addressSpaceOf(options.image);
    this.memory = new WebAssembly.Memory({ initial: options.pages ?? options.image.pages });
    this.imports = {
      env: {
        memory: this.memory,
        syscall: (number: bigint, a1: bigint, a2: bigint, a3: bigint): bigint =>
          this.#syscall(number, a1, a2, a3),
      },
    };
  }

  /** Runs a translated module and reports what it produced. */
  run(module: Uint8Array): Outcome {
    const instance = new WebAssembly.Instance(new WebAssembly.Module(module), this.imports);
    const entry = Object.values(instance.exports)[0];
    if (typeof entry !== "function") {
      throw new Error("the module has no exported function to run");
    }
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

  #syscall(number: bigint, a1: bigint, a2: bigint, a3: bigint): bigint {
    if (number === SYSCALLS.write) return this.#write(Number(a1), a2, Number(a3));
    if (number === SYSCALLS.exit || number === SYSCALLS.exitGroup) {
      const code = Number(a1) & 0xff;
      this.#exitCode = code;
      throw new Exit(code);
    }
    return ERRNO.ENOSYS;
  }

  #write(descriptor: number, address: bigint, count: number): bigint {
    if (descriptor !== 1 && descriptor !== 2) return ERRNO.EBADF;
    if (count < 0) return ERRNO.EINVAL;
    const bytes = this.bytesAt(address, count);
    if (bytes === undefined) return ERRNO.EFAULT;
    (descriptor === 1 ? this.#stdout : this.#stderr).push(...bytes);
    return BigInt(count);
  }

  /** The bytes at a virtual address, or `undefined` when the range is not entirely inside the image. */
  bytesAt(address: bigint, count: number): Uint8Array | undefined {
    if (count === 0) return new Uint8Array(0);
    const offset = this.addresses.offsetOf(address);
    if (offset === undefined || offset + count > this.memory.buffer.byteLength) return undefined;
    return new Uint8Array(this.memory.buffer, offset, count);
  }
}
