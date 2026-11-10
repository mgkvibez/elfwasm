/**
 * Memory, against the processor rather than against this translator.
 *
 * `tests/memory.test.ts` checks the instructions this repository chose to write. This checks the whole path a real program
 * takes: assemble, link at the addresses the linker picks, read the resulting executable's own program headers into a
 * WASM memory, translate the bytes the symbol table points at, and run. The processor runs the same executable. If the two
 * disagree about where a store landed, nothing else in this repository would have noticed.
 *
 * It skips when binutils is absent, for the same reason the other agreement test does: a cold checkout has no toolchain.
 */

import { describe, expect, it } from "vitest";

// @ts-expect-error node:child_process is intentionally outside this project's type surface
const { execFileSync } = await import("node:child_process");
// @ts-expect-error node:fs is intentionally outside this project's type surface
const { mkdtempSync, readFileSync, rmSync } = await import("node:fs");
// @ts-expect-error node:os is intentionally outside this project's type surface
const { tmpdir } = await import("node:os");
// @ts-expect-error node:path is intentionally outside this project's type surface
const { join } = await import("node:path");

import { translateBytes } from "../src/translate.js";
import { addressSpaceOf, createMemory, imageOf, loadImage } from "../src/image.js";
import type { MemoryImage } from "../src/image.js";

const START = `.globl _start
_start:
    call target
    mov %rax, %rdi
    mov $60, %rax
    syscall
`;

const runtime = (
  globalThis as unknown as {
    WebAssembly: {
      Memory: new (descriptor: { initial: number }) => { readonly buffer: ArrayBuffer };
      Module: new (bytes: Uint8Array) => unknown;
      Instance: new (
        module: unknown,
        imports: { env: { memory: { readonly buffer: ArrayBuffer } } },
      ) => { exports: Record<string, () => bigint> };
    };
  }
).WebAssembly;

function has(binary: string): boolean {
  try {
    execFileSync("sh", ["-c", `command -v ${binary}`], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function status(executable: string): number {
  try {
    execFileSync(executable, [], { stdio: "ignore" });
    return 0;
  } catch (error) {
    const code = (error as { status?: unknown }).status;
    if (typeof code === "number") return code;
    throw error;
  }
}

/** A symbol's address and size from `nm -S`, which is how the test finds the bytes it must translate. */
function symbol(executable: string, name: string): { address: bigint; size: number } {
  const listed = execFileSync("nm", ["-S", executable], { encoding: "utf8" }) as string;
  for (const line of listed.split("\n")) {
    const fields = line.trim().split(/\s+/);
    // `nm -S` prints address, size, type, name for a symbol that has a size
    if (fields.length >= 4 && fields[3] === name) {
      return { address: BigInt(`0x${fields[0]!}`), size: Number.parseInt(fields[1]!, 16) };
    }
  }
  throw new Error(`no sized symbol named ${name} in ${executable}`);
}

/** The file offset of a virtual address, so the code's own bytes can be cut out of the executable. */
function fileOffsetOf(image: MemoryImage, address: bigint): number {
  for (const segment of image.segments) {
    if (address >= segment.virtualAddress && address < segment.virtualAddress + segment.fileSize) {
      return Number(segment.fileOffset + (address - segment.virtualAddress));
    }
  }
  throw new Error(`0x${address.toString(16)} is not in a file-backed segment`);
}

/**
 * The programs, and the number the processor should produce for each.
 *
 * Writing the expected value here rather than deriving it means both sides are checked against a third thing: a program
 * whose assembler output happened to be wrong would still disagree with a wrong expectation, but a translation that
 * agreed with a wrong assembler would not go unnoticed the way it would if the expectation came from the translator.
 */
const PROGRAMS = [
  {
    name: "memory",
    expected: 0x2a,
    memory: { symbol: "slot", bytes: [0x2a, 0, 0, 0, 0, 0, 0, 0] },
  },
  {
    name: "addressing",
    expected: 7 + 5,
    memory: { symbol: "slot", bytes: [7, 0, 0, 0, 0, 0, 0, 0] },
  },
  {
    name: "stack",
    expected: 10,
    stackPages: 2,
  },
] as const;

const toolchain = ["as", "ld", "nm"].every(has);

describe.skipIf(!toolchain)("memory against the processor, on a linked binary", () => {
  for (const program of PROGRAMS) {
    it(`agrees with the processor about ${program.name}, through an image built from that binary`, () => {
      const directory = mkdtempSync(join(tmpdir(), `elfwasm-${program.name}-`));
      try {
        const startPath = join(directory, "start.s");
        execFileSync("sh", ["-c", `cat > ${startPath} <<'EOF'\n${START}EOF`]);
        const startObject = join(directory, "start.o");
        const programObject = join(directory, "program.o");
        const executable = join(directory, program.name);

        execFileSync("as", ["--64", "-o", startObject, startPath]);
        execFileSync("as", ["--64", "-o", programObject, `tools/native/memory/${program.name}.s`]);
        // -no-pie because these programs address their data symbols absolutely, which is the addressing being translated
        execFileSync("ld", ["-nostdlib", "-no-pie", "-o", executable, startObject, programObject]);
        execFileSync("chmod", ["+x", executable]);

        const native = status(executable);

        // the executable's own program headers decide where things are, so nothing here hardcodes an address
        const bytes = Uint8Array.from(readFileSync(executable) as Uint8Array);
        const image = imageOf(bytes);
        const addresses = addressSpaceOf(image);
        const code = symbol(executable, "target");

        // a program that uses the stack needs pages above the image, which the caller allocates, not the module
        const stackPages = "stackPages" in program ? program.stackPages : 0;
        const pages = image.pages + stackPages;
        const memory =
          stackPages === 0 ? createMemory(image) : new runtime.Memory({ initial: pages });
        loadImage(bytes, image, memory);
        const buffer = new Uint8Array(memory.buffer);
        const codeStart = fileOffsetOf(image, code.address);
        const module = translateBytes(bytes.subarray(codeStart, codeStart + code.size), "main", {
          addresses,
          codeAddress: code.address,
          ...(stackPages === 0 ? { memoryPages: image.pages } : { memoryPages: pages, stackPages }),
        });
        const instance = new runtime.Instance(new runtime.Module(module), { env: { memory } });
        const translated = Number(instance.exports.main!());


        expect(native).toBe(program.expected);
        expect(translated & 0xff).toBe(native);

        // a program with a data symbol also checks where its store landed, in the caller's memory rather than in a report
        if ("memory" in program) {
          const slot = symbol(executable, program.memory.symbol);
          const offset = addresses.offsetOf(slot.address);
          expect(offset).toBeTypeOf("number");
          expect(Array.from(buffer.subarray(offset!, offset! + 8))).toEqual([...program.memory.bytes]);
        }
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    });
  }
});
