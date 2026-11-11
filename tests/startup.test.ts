/**
 * `R_X86_64_IRELATIVE`: the relocation whose value is whatever a function returns.
 *
 * The proof is a hand-linked PIE whose `.data` holds an ifunc symbol rather than an address. The linker turns that into an
 * IRELATIVE relocation whose addend is the *resolver's* address, and the resolver returns the address of a marker that only it
 * knows, because it computes it with `lea marker(%rip)`. If the loader wrote the resolver's address instead, or wrote a
 * link-time one, the program would compare two addresses that differ; if it wrote the resolver's answer, they agree.
 *
 * The same bytes run at two different bases, so the answer has to be a *runtime* address rather than a constant that happened
 * to match: the resolver ran where the image was placed, and its answer moved with it. Run natively the same bytes fail,
 * because a binary linked with no dynamic linker has nobody to resolve its ifuncs, and that is the differential: 1 without the
 * relocations applied, 0 with them.
 */

import { describe, expect, it } from "vitest";

import type { MemoryImage } from "../src/image.js";
import { loadResolved, translatedResolver } from "../src/startup.js";
import { Machine } from "../src/syscalls.js";
import { translateProgram } from "../src/translate.js";

// @ts-expect-error node:child_process is intentionally outside this project's type surface
const { execFileSync } = await import("node:child_process");
// @ts-expect-error node:fs is intentionally outside this project's type surface
const { mkdtempSync, readFileSync, rmSync, writeFileSync } = await import("node:fs");
// @ts-expect-error node:os is intentionally outside this project's type surface
const { tmpdir } = await import("node:os");
// @ts-expect-error node:path is intentionally outside this project's type surface
const { join } = await import("node:path");

const has = (binary: string): boolean => {
  try {
    execFileSync("sh", ["-c", `command -v ${binary}`], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
};

const toolchain = ["as", "ld", "readelf", "nm"].every(has);

/**
 * A PIE whose slot is an ifunc: the loader has to call the resolver to know what belongs there.
 *
 * `slot` is `.quad resolver`, and `resolver` is declared `@gnu_indirect_function`, which is the assembly spelling of an ifunc
 * symbol. The linker cannot fold that to a constant, because the value is only known once the function has returned.
 */
const SOURCE = `.globl _start
_start:
    mov slot(%rip), %rax        # what the loader wrote, which is the resolver's answer
    lea marker(%rip), %rbx      # what it should be, computed independently here
    cmp %rbx, %rax
    jne wrong
    xor %rdi, %rdi
    mov $60, %rax
    syscall
wrong:
    mov $1, %rdi
    mov $60, %rax
    syscall
.size _start, .-_start

.type resolver, @gnu_indirect_function
resolver:
    lea marker(%rip), %rax
    ret
.size resolver, .-resolver

.data
slot:
    .quad resolver
marker:
    .byte 0x42
`;

/** A symbol's link-time address and size, which for a PIE are offsets from zero rather than load addresses. */
function symbol(executable: string, name: string): { address: bigint; size: number } {
  const listed = execFileSync("nm", ["-S", executable], { encoding: "utf8" }) as string;
  for (const line of listed.split("\n")) {
    const fields = line.trim().split(/\s+/);
    // a data symbol with no `.size` gets no size column at all, so both shapes have to be read
    if (fields.length === 4 && fields[3] === name) {
      return { address: BigInt(`0x${fields[0]!}`), size: Number.parseInt(fields[1]!, 16) };
    }
    if (fields.length === 3 && fields[2] === name) return { address: BigInt(`0x${fields[0]!}`), size: 0 };
  }
  throw new Error(`no symbol named ${name} in ${executable}`);
}

/** Where a virtual address sits in the file, which is how a symbol's bytes are cut out for the translator. */
function fileIndexOf(image: MemoryImage, address: bigint): number {
  for (const segment of image.segments) {
    if (address >= segment.virtualAddress && address < segment.virtualAddress + segment.fileSize) {
      return Number(segment.fileOffset) + Number(address - segment.virtualAddress);
    }
  }
  throw new Error(`no segment maps 0x${address.toString(16)}`);
}

/** Where a virtual address sits in the loaded memory, which is how a value the loader wrote is read back. */
function wasmIndexOf(image: MemoryImage, address: bigint): number {
  for (const segment of image.segments) {
    if (address >= segment.virtualAddress && address < segment.virtualAddress + segment.memorySize) {
      return segment.wasmAddress + Number(address - segment.virtualAddress);
    }
  }
  throw new Error(`no segment maps 0x${address.toString(16)}`);
}

interface Built {
  readonly executable: string;
  readonly bytes: Uint8Array;
  readonly code: { readonly address: bigint; readonly size: number };
}

function build(directory: string): Built {
  const source = join(directory, "ifunc.s");
  writeFileSync(source, SOURCE);
  const object = join(directory, "ifunc.o");
  const executable = join(directory, "ifunc");
  execFileSync("as", ["--64", "-o", object, source], { stdio: "ignore" });
  execFileSync("ld", ["-pie", "--no-dynamic-linker", "-o", executable, object], { stdio: "ignore" });
  return { executable, bytes: readFileSync(executable), code: symbol(executable, "_start") };
}

/** Translates `_start` out of the loaded image and runs it in the loader's own memory. */
function runAt(built: Built, base: bigint): number {
  const loaded = loadResolved(built.bytes, { base });
  const machine = new Machine({ image: loaded.image, memory: loaded.memory });
  const entry = base + built.code.address;
  const start = fileIndexOf(loaded.image, entry);
  const module = translateProgram(built.bytes.subarray(start, start + built.code.size), "main", {
    addresses: machine.addresses,
    codeAddress: entry,
    code: { bytes: built.bytes, virtualAddress: loaded.image.baseAddress },
    memoryPages: machine.memory.buffer.byteLength / 0x10000,
    syscalls: true,
  });
  return machine.run(module).exitCode;
}

describe.skipIf(!toolchain)("an ifunc, resolved by the image's own code", () => {
  it("puts the resolver's answer in the slot, and the program agrees at two bases", () => {
    const directory = mkdtempSync(join(tmpdir(), "elfwasm-ifunc-"));
    try {
      const built = build(directory);

      // the relocation really is an ifunc, which is the whole premise of the test
      const relocations = execFileSync("readelf", ["-r", built.executable], { encoding: "utf8" }) as string;
      expect(relocations).toContain("R_X86_64_IRELATIV");

      // zero at two bases: the slot holds what the resolver returned, which is the marker's runtime address
      expect(runAt(built, 0x400000n)).toBe(0);
      // and at a different base, so the answer cannot have been a link-time constant
      expect(runAt(built, 0x800000n)).toBe(0);

      /*
       * Linux is not the reference here, and that is worth pinning rather than hiding: this binary is linked with
       * `--no-dynamic-linker`, so nothing resolves its IRELATIVE relocations. The kernel maps the segments and jumps to the
       * entry with the slot still holding the resolver's link-time address, so the program compares and fails, natively, with
       * status 1. The translated run is the one that resolves the ifunc, and the differential is the two statuses: 1 without
       * relocations applied, 0 with them.
       */
      let nativeExit = 0;
      try {
        execFileSync(built.executable, [], { stdio: "ignore" });
      } catch (error) {
        nativeExit = (error as { status?: number }).status ?? -1;
      }
      expect(nativeExit).toBe(1);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("writes the resolver's answer into the slot, not the resolver's own address", () => {
    const directory = mkdtempSync(join(tmpdir(), "elfwasm-ifunc-"));
    try {
      const built = build(directory);
      const base = 0x400000n;
      const loaded = loadResolved(built.bytes, { base });
      const written = new DataView(loaded.memory.buffer).getBigUint64(
        wasmIndexOf(loaded.image, base + symbol(built.executable, "slot").address),
        true,
      );
      // the marker's address: the answer the resolver computed, at this base
      expect(written).toBe(base + symbol(built.executable, "marker").address);
      // and specifically not the resolver's address, which is what a loader that copied the addend would write
      expect(written).not.toBe(base + symbol(built.executable, "resolver").address);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});

describe.skipIf(!has("gcc") || !has("readelf"))("the ifuncs a real glibc static PIE carries", () => {
  it("refuses at the first resolver, naming the address and the instruction it cannot translate", () => {
    const directory = mkdtempSync(join(tmpdir(), "elfwasm-glibc-ifunc-"));
    try {
      const source = join(directory, "main.c");
      const executable = join(directory, "pie");
      writeFileSync(source, "static int counter; int main(void) { counter = counter + 7; return counter; }\n");
      execFileSync("gcc", ["-static-pie", "-o", executable, source], { stdio: "ignore" });

      /*
       * glibc's own resolvers are longer than the ones in the subset: the first one is reached at 0x420500 and holds an
       * instruction the decoder does not have yet, so this is a refusal about the *decoder*, not about the mechanism. Pinning
       * the message means the next piece of decoder work shows up here as a change of address rather than as silence.
       *
       * It has moved once already: the byte forms of group 3 and of the register comparisons carried it from the resolver's
       * fourth instruction to its seventeenth, where it now wants `cmovne`, which is a two-byte opcode.
       *
       * Where it lands depends on the machine's glibc: the CI image's build ends its resolver earlier, with a jump out of the
       * resolver's own bytes into a vectorised copy routine. Pinning that exact instruction would pin the runner's libc rather
       * than this repository, so what is pinned is that the refusal is a named one either way.
       */
      let refusal: unknown;
      try {
        loadResolved(readFileSync(executable) as Uint8Array, { base: 0x400000n });
      } catch (error) {
        refusal = error;
      }
      // a named refusal carrying a location, from the translator or the loader, rather than a crash or an answer computed from
      // code that was only half understood
      expect(refusal).toBeInstanceOf(Error);
      expect((refusal as Error).message).toMatch(/^(a resolver at 0x[0-9a-f]+|at offset 0x[0-9a-f]+)/);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});

describe("the extent a resolver is translated from", () => {
  /** One page of image at the origin, holding only the bytes the test supplied. */
  const imageOf = (length: number): MemoryImage => ({
    baseAddress: 0x400000n,
    loadBias: 0n,
    wasmBytes: 0x10000,
    pages: 1,
    entryAddress: 0,
    segments: [
      {
        virtualAddress: 0x400000n,
        fileOffset: 0n,
        wasmAddress: 0,
        fileSize: BigInt(length),
        memorySize: 0x10000n,
        permissions: { read: true, write: true, execute: true },
      },
    ],
  });

  const runtime = (
    globalThis as unknown as {
      WebAssembly: { Memory: new (descriptor: { initial: number }) => { readonly buffer: ArrayBuffer } };
    }
  ).WebAssembly;

  const resolve = (code: number[], limit?: number): bigint => {
    const bytes = Uint8Array.from(code);
    // two pages: the image, and the page the stack sits above it
    const memory = new runtime.Memory({ initial: 2 });
    new Uint8Array(memory.buffer).set(bytes, 0);
    return translatedResolver({
      bytes,
      image: imageOf(bytes.length),
      memory: memory as unknown as { readonly buffer: ArrayBuffer },
      stackPages: 1,
      ...(limit === undefined ? {} : { limit }),
    })(0x400000n);
  };

  it("runs a straight-line resolver and answers with %rax", () => {
    // mov $0x1234, %rax ; ret
    expect(resolve([0x48, 0xc7, 0xc0, 0x34, 0x12, 0x00, 0x00, 0xc3])).toBe(0x1234n);
  });

  it("does not stop at a 0xc3 that is part of an immediate", () => {
    // mov $0xc3, %rax ; add $1, %rax ; ret  -- the first 0xc3 is data, and the answer proves the walk skipped it
    expect(resolve([0x48, 0xc7, 0xc0, 0xc3, 0x00, 0x00, 0x00, 0x48, 0x83, 0xc0, 0x01, 0xc3])).toBe(0xc4n);
  });

  it("refuses a resolver with no ret in reach, by name and with the limit it searched", () => {
    // mov $1, %rax and nothing else, padded so the decoder has bytes to walk
    const code = [0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, ...Array.from({ length: 32 }, () => 0x90)];
    expect(() => resolve(code, 32)).toThrow(/has no ret in its first 32 bytes/);
  });

  it("follows a jump out of the resolver's own bytes to the ret that ends the answer", () => {
    /*
     * A resolver is allowed to end in a tail call: the function it jumps into returns the address the resolver would have
     * returned, so its `ret` is the resolver's ret. A linear scan for `ret` would take the bytes between the two as the
     * body and then find the jump's target between two instructions; following the jump decodes exactly what is reachable.
     *
     * `mov $0x2a, %eax; jmp +5` lands on `mov $7, %eax` at offset 12, five bytes past the padding the jump skips.
     */
    expect(resolve([
      0xb8, 0x2a, 0x00, 0x00, 0x00,
      0xeb, 0x05,
      0x90, 0x90, 0x90, 0x90, 0x90,
      0xb8, 0x07, 0x00, 0x00, 0x00,
      0xc3,
    ])).toBe(7n);
  });

  it("translates a nop and a prefetch as nothing at all, memory operand or not", () => {
    // `nop (%rax); prefetch (%rax); mov $5, %eax; ret`: padding must not need the image mapped, because it reads no memory
    expect(resolve([0x0f, 0x1f, 0x00, 0x0f, 0x18, 0x08, 0xb8, 0x05, 0x00, 0x00, 0x00, 0xc3])).toBe(5n);
  });

  it("keeps a comparison readable across a lea, which computes an address and writes no flag", () => {
    /*
     * `mov $4,%eax; cmp $4,%eax; lea 4(%rax),%rdx; cmove %rbx,%rax; ret`. glibc's own ifunc resolvers are this shape: a test,
     * two `lea`s that pick the two candidate routines, and a conditional move between them. Treating `lea` as though it clobbered
     * the flags is what stopped translation of the real one at its first `cmovne`.
     */
    expect(resolve([
      0xb8, 0x04, 0x00, 0x00, 0x00,
      0x48, 0xc7, 0xc3, 0x55, 0x00, 0x00, 0x00,
      0x83, 0xf8, 0x04,
      0x48, 0x8d, 0x50, 0x04,
      0x48, 0x0f, 0x44, 0xc3,
      0xc3,
    ])).toBe(0x55n);
  });

  it("keeps a comparison readable across a push and a pop, which move memory and no flag", () => {
    // `mov $0x55,%rbx; mov $9,%eax; cmp $9,%eax; push %rax; pop %rax; cmove %rbx,%rax; ret`
    expect(resolve([
      0x48, 0xc7, 0xc3, 0x55, 0x00, 0x00, 0x00,
      0xb8, 0x09, 0x00, 0x00, 0x00,
      0x83, 0xf8, 0x09,
      0x50,
      0x58,
      0x48, 0x0f, 0x44, 0xc3,
      0xc3,
    ])).toBe(0x55n);
  });

  it("compares a byte of memory, which is how a resolver reads a feature flag", () => {
    /*
     * `testb $0x2, 0x10(%rip)` then `jne`: the byte it reads is at offset 0x17 of the image, and 0x02 & 0x02 is not zero, so the
     * branch is taken and the answer is 5 rather than 1. glibc's resolvers test `_dl_x86_cpu_features` exactly this way, and a
     * byte comparison was refused whenever either operand lived in memory.
     */
    const code = [
      0xf6, 0x05, 0x10, 0x00, 0x00, 0x00, 0x02, // testb $0x2, 0x10(%rip)
      0xb8, 0x05, 0x00, 0x00, 0x00, // mov $5, %eax
      0x75, 0x05, // jne +5
      0xb8, 0x01, 0x00, 0x00, 0x00, // mov $1, %eax
      0xc3, // ret
      0x00, 0x00, 0x00, // padding up to the flag the test reads
      0x02,
    ];
    expect(resolve(code)).toBe(5n);
    // with the bit clear the branch is not taken, so the same code answers 1
    expect(resolve([...code.slice(0, 0x17), 0x01])).toBe(1n);
  });

  it("refuses an address that is not in an executable file-backed segment", () => {
    const bytes = Uint8Array.from([0xc3]);
    const memory = new runtime.Memory({ initial: 1 });
    // zero is below the image, so there is no code there to walk
    expect(() =>
      translatedResolver({
        bytes,
        image: imageOf(1),
        memory: memory as unknown as { readonly buffer: ArrayBuffer },
      })(0n),
    ).toThrow(/not inside any file-backed segment/);
  });
});
