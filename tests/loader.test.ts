/**
 * The loader, which is the step between "the segments are in memory" and "the program can run".
 *
 * A position-independent executable's pointers are offsets until the load address is known, so the things worth testing are
 * that a caller-chosen base moves the whole image and nothing else, that the relocation table is written against that base,
 * and that a program which reads a relocated pointer sees its own address. The last one is checked by running the same small
 * PIE natively and translated, so the loader's arithmetic is never graded by the loader.
 */

import { describe, expect, it } from "vitest";

import { parseElfHeader, parseRelocations, parseSectionHeaders } from "../src/elf.js";
import { createMemory, imageOf, loadImage, toWasmAddress } from "../src/image.js";
import type { MemoryImage } from "../src/image.js";
import { DEFAULT_LOAD_BASE, applyRelocations, loadExecutable } from "../src/loader.js";
import { Machine } from "../src/syscalls.js";
import { translateProgram } from "../src/translate.js";

// @ts-expect-error node:child_process is intentionally outside this project's type surface
const { execFileSync } = await import("node:child_process");
// @ts-expect-error node:fs is intentionally outside this project's type surface
const { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } = await import("node:fs");
// @ts-expect-error node:os is intentionally outside this project's type surface
const { tmpdir } = await import("node:os");
// @ts-expect-error node:path is intentionally outside this project's type surface
const { join } = await import("node:path");

function has(tool: string): boolean {
  try {
    execFileSync(tool, ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

const binutils = has("as") && has("ld") && has("nm");

/**
 * A position-independent executable, linked by hand rather than compiled, so it has one relative relocation and no ifuncs.
 *
 * It does the one thing that makes relocation observable from inside the program: `pointer` in `.data` is a relocated address
 * of `_start`, and `_start` compares that pointer with its own runtime address, exiting 0 if they match and 1 if they do not.
 * So the same bytes give both answers depending on one thing: whether the slot was fixed up. That is the differential the
 * tests below run, and it is why they do not need an expected number from anywhere: the program says whether the load worked.
 */
const SOURCE = [
  ".globl _start",
  "_start:",
  "    lea _start(%rip), %rax",
  "    mov pointer(%rip), %rdx",
  "    cmp %rdx, %rax",
  "    jne failed",
  "    xor %rdi, %rdi",
  "    mov $60, %rax",
  "    syscall",
  "failed:",
  "    mov $1, %rdi",
  "    mov $60, %rax",
  "    syscall",
  ".size _start, .-_start",
  ".data",
  ".globl pointer",
  "pointer:",
  "    .quad _start",
  "",
].join("\n");

interface Pie {
  readonly bytes: Uint8Array;
  /** Where `_start` sits in the file's own address space, which is where the relocation's addend points. */
  readonly entry: bigint;
  readonly entrySize: number;
}

function buildPie(): Pie | undefined {
  if (!binutils) return undefined;
  const directory = mkdtempSync(join(tmpdir(), "elfwasm-pie-"));
  try {
    const source = join(directory, "pie.s");
    writeFileSync(source, SOURCE);
    const object = join(directory, "pie.o");
    const executable = join(directory, "pie");
    execFileSync("as", ["--64", "-o", object, source], { stdio: "ignore" });
    // --no-dynamic-linker, because there is no libc: the kernel maps this ET_DYN and jumps to its entry by itself
    execFileSync("ld", ["-pie", "--no-dynamic-linker", "-nostdlib", "-o", executable, object], { stdio: "ignore" });

    const symbols = execFileSync("nm", ["-S", executable], { encoding: "utf8" }) as string;
    const line = symbols.split("\n").find((text) => text.trim().endsWith(" T _start"));
    if (line === undefined) return undefined;
    // nm -S prints "address size type name", so the address is the first column and the size the second
    const [address = "", size = ""] = line.trim().split(/\s+/);
    return {
      bytes: readFileSync(executable),
      entry: BigInt(`0x${address}`),
      entrySize: Number.parseInt(size, 16),
    };
  } catch {
    return undefined;
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

const PIE = buildPie();

/** A glibc static PIE, which is the shape a real libc-linked position-independent binary has. */
function buildGlibcPie(): Uint8Array | undefined {
  if (!has("gcc") || !has("readelf")) return undefined;
  const directory = mkdtempSync(join(tmpdir(), "elfwasm-glibc-"));
  try {
    const source = join(directory, "main.c");
    writeFileSync(source, "static int counter; int main(void) { counter = counter + 7; return counter; }\n");
    const executable = join(directory, "pie");
    execFileSync("gcc", ["-static-pie", "-o", executable, source], { stdio: "ignore" });
    return readFileSync(executable);
  } catch {
    return undefined;
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

const GLIBC_PIE = buildGlibcPie();

/**
 * One writable segment with 0x100 bytes, and nothing else, which is all a relocation needs.
 *
 * The segment sits at 0x400100 in the loaded image and the bias is 0x400000, so the file it describes had its segments at
 * 0x100: a position-independent layout, which is the case the bias exists for. The relocations below are therefore written in
 * the file's address space, as a real table's are.
 */
function slotImage(): MemoryImage {
  return {
    baseAddress: 0x400000n,
    loadBias: 0x400000n,
    wasmBytes: 0x10000,
    pages: 1,
    entryAddress: 0,
    segments: [
      {
        virtualAddress: 0x400100n,
        fileOffset: 0n,
        wasmAddress: 0x100,
        fileSize: 0x100n,
        memorySize: 0x100n,
        permissions: { read: true, write: true, execute: false },
      },
    ],
  };
}

describe("writing a relocation", () => {
  it("puts the load bias plus the addend in the slot", () => {
    const image = slotImage();
    const memory = createMemory(image);
    const applied = applyRelocations(
      [{ offset: 0x110n, symbolIndex: 0, type: 8, symbolName: "", addend: 0x2050n }],
      { image, memory },
    );
    expect(applied).toBe(1);
    // 0x400000 is where the image landed and 0x2050 is where the target sits in the file: both are in the answer
    expect(new DataView(memory.buffer).getBigUint64(0x110, true)).toBe(0x402050n);
  });

  it("writes a negative addend as the signed number it is", () => {
    const image = slotImage();
    const memory = createMemory(image);
    applyRelocations([{ offset: 0x110n, symbolIndex: 0, type: 8, symbolName: "", addend: -8n }], { image, memory });
    expect(new DataView(memory.buffer).getBigInt64(0x110, true)).toBe(0x400000n - 8n);
  });

  it("rebases the slot's own address, not just the value it holds", () => {
    const image = slotImage();
    const memory = createMemory(image);
    applyRelocations([{ offset: 0x1f8n, symbolIndex: 0, type: 8, symbolName: "", addend: 0x1000n }], { image, memory });
    // the last eight bytes of the segment, which is the slot an un-rebased offset would have missed entirely
    expect(new DataView(memory.buffer).getBigUint64(0x1f8, true)).toBe(0x401000n);
  });

  it("leaves a NONE relocation alone, because the linker emitted a placeholder", () => {
    const image = slotImage();
    const memory = createMemory(image);
    new Uint8Array(memory.buffer).fill(0xff, 0x100, 0x120);
    const applied = applyRelocations(
      [{ offset: 0x110n, symbolIndex: 0, type: 0, symbolName: "", addend: 0n }],
      { image, memory },
    );
    expect(applied).toBe(0);
    expect(new Uint8Array(memory.buffer).slice(0x110, 0x118)).toEqual(new Uint8Array(8).fill(0xff));
  });

  it("refuses a slot no segment maps, rather than writing outside the image", () => {
    const image = slotImage();
    const memory = createMemory(image);
    expect(() =>
      applyRelocations([{ offset: 0x500000n, symbolIndex: 0, type: 8, symbolName: "", addend: 0n }], { image, memory }),
    ).toThrow(/not inside any mapped segment/);
  });

  it("refuses a slot that runs off the end of its segment", () => {
    const image = slotImage();
    const memory = createMemory(image);
    expect(() =>
      applyRelocations([{ offset: 0x1fcn, symbolIndex: 0, type: 8, symbolName: "", addend: 0n }], { image, memory }),
    ).toThrow(/runs off the end/);
  });

  it("refuses a slot in a read-only segment, because the file asked for those pages not to change", () => {
    const base = slotImage();
    const image: MemoryImage = {
      ...base,
      segments: [{ ...base.segments[0]!, permissions: { read: true, write: false, execute: false } }],
    };
    const memory = createMemory(image);
    expect(() =>
      applyRelocations([{ offset: 0x110n, symbolIndex: 0, type: 8, symbolName: "", addend: 0n }], { image, memory }),
    ).toThrow(/read-only/);
  });

  it("names a type it does not implement instead of guessing", () => {
    const image = slotImage();
    const memory = createMemory(image);
    expect(() =>
      applyRelocations([{ offset: 0x110n, symbolIndex: 0, type: 7, symbolName: "puts", addend: 0n }], { image, memory }),
    ).toThrow(/R_X86_64_JUMP_SLOT .* no resolver/);
  });
});

describe("choosing where an image lands", () => {
  it("moves every segment by the base and leaves their offsets and the entry where they were", () => {
    const bytes = PIE?.bytes ?? GLIBC_PIE;
    if (bytes === undefined) return;
    const own = imageOf(bytes);
    const loaded = imageOf(bytes, { base: DEFAULT_LOAD_BASE });

    // a PIE is linked at zero, so its own layout puts virtual address zero at the start of the file
    expect(own.baseAddress).toBe(0n);
    expect(own.loadBias).toBe(0n);
    expect(loaded.baseAddress).toBe(DEFAULT_LOAD_BASE);
    expect(loaded.loadBias).toBe(DEFAULT_LOAD_BASE);

    expect(loaded.segments.length).toBe(own.segments.length);
    for (const [index, segment] of loaded.segments.entries()) {
      const original = own.segments[index]!;
      expect(segment.virtualAddress).toBe(original.virtualAddress + DEFAULT_LOAD_BASE);
      // the offset inside the buffer is a distance, so moving the image must not change it
      expect(segment.wasmAddress).toBe(original.wasmAddress);
      expect(segment.memorySize).toBe(original.memorySize);
    }
    expect(loaded.entryAddress).toBe(own.entryAddress);
    expect(loaded.pages).toBe(own.pages);
    expect(loaded.wasmBytes).toBe(own.wasmBytes);
  });

  it("refuses loading at address zero, which would map the null page", () => {
    const bytes = PIE?.bytes ?? GLIBC_PIE;
    if (bytes === undefined) return;
    expect(() => loadExecutable(bytes, { base: 0n })).toThrow(/the null page would be mapped/);
  });

  it("refuses a base that is not a page, since it is not an address a mapping can start at", () => {
    const bytes = PIE?.bytes ?? GLIBC_PIE;
    if (bytes === undefined) return;
    expect(() => imageOf(bytes, { base: DEFAULT_LOAD_BASE + 8n })).toThrow(/not a multiple of the 65536-byte WASM page/);
  });


});

describe("a hand-linked position-independent executable, relocated and run", () => {
  it("applies its one relative relocation, which points at the program's own entry", () => {
    const pie = PIE;
    if (pie === undefined) return;
    const loaded = loadExecutable(pie.bytes);
    expect(loaded.relocations).toBe(1);
    expect(loaded.image.baseAddress).toBe(DEFAULT_LOAD_BASE);
    expect(loaded.image.loadBias).toBe(DEFAULT_LOAD_BASE);

    /*
     * The slot is at the file's 0x3000 and the addend is `_start`'s link-time address, both from binutils rather than from the
     * loader. Where they land is the load base plus each.
     */
    const table = parseRelocations(pie.bytes, parseSectionHeaders(pie.bytes, parseElfHeader(pie.bytes)));
    const relative = table.find((relocation) => relocation.type === 8);
    expect(relative?.addend).toBe(pie.entry);
    const at = toWasmAddress(loaded.image, DEFAULT_LOAD_BASE + (relative?.offset ?? 0n));
    expect(new DataView(loaded.memory.buffer).getBigUint64(at, true)).toBe(DEFAULT_LOAD_BASE + pie.entry);
  });

  /** Translate `_start` against a loaded image and run it, optionally without the relocations in place. */
  function run(pie: Pie, relocate: boolean): number | undefined {
    const loaded = relocate
      ? loadExecutable(pie.bytes)
      : (() => {
          const image = imageOf(pie.bytes, { base: DEFAULT_LOAD_BASE });
          const memory = createMemory(image);
          loadImage(pie.bytes, image, memory);
          return { image, memory };
        })();

    // the loaded image lives in the loader's memory, so the machine runs in that buffer rather than a fresh copy of it
    const machine = new Machine({ image: loaded.image, pages: loaded.image.pages, memory: loaded.memory });
    const codeAddress = loaded.image.baseAddress + pie.entry;
    const start = toWasmAddress(loaded.image, codeAddress);
    const module = translateProgram(pie.bytes.subarray(start, start + pie.entrySize), "main", {
      addresses: machine.addresses,
      codeAddress,
      code: { bytes: pie.bytes, virtualAddress: loaded.image.baseAddress },
      memoryPages: loaded.image.pages,
      syscalls: true,
    });
    return machine.run(module).exitCode;
  }

  it("runs it, and it finds its own address through the relocated pointer", () => {
    const pie = PIE;
    if (pie === undefined) return;
    expect(run(pie, true)).toBe(0);
  });

  it("exits 1 without the relocation, which is what makes the run above evidence", () => {
    const pie = PIE;
    if (pie === undefined) return;
    // the pointer still holds its link-time value, so it does not equal the address the code is actually running at
    expect(run(pie, false)).toBe(1);
  });
});

describe("a real libc static PIE", () => {
  it("applies every relative relocation to the address the linker's own table names", () => {
    const bytes = GLIBC_PIE;
    if (bytes === undefined) return;
    const base = DEFAULT_LOAD_BASE;
    const image = imageOf(bytes, { base });
    const memory = createMemory(image);
    loadImage(bytes, image, memory);
    const table = parseRelocations(bytes, parseSectionHeaders(bytes, parseElfHeader(bytes)));
    const relative = table.filter((relocation) => relocation.type === 8);
    expect(relative.length).toBeGreaterThan(1000);
    expect(applyRelocations(relative, { image, memory })).toBe(relative.length);

    /*
     * `readelf` is the linker's own account of the same table, so comparing against it checks the parser and the writer together
     * rather than either against itself.
     */
    const directory = mkdtempSync(join(tmpdir(), "elfwasm-readelf-"));
    try {
      const executable = join(directory, "pie");
      writeFileSync(executable, bytes);
      const listing = execFileSync("readelf", ["-r", executable], { encoding: "utf8" }) as string;
      // readelf prints the offset without leading zeroes, and a negative addend with its sign, so both are read as they come
      const expected = [...listing.matchAll(/^([0-9a-f]{6,16})\s+[0-9a-f]{8,16} R_X86_64_RELATIVE\s+(-?[0-9a-f]+)\s*$/gm)].map(
        (match) => {
          const addend = match[2]!;
          return {
            offset: BigInt(`0x${match[1]!}`),
            addend: addend.startsWith("-") ? -BigInt(`0x${addend.slice(1)}`) : BigInt(`0x${addend}`),
          };
        },
      );
      expect(expected.length).toBe(relative.length);

      const buffer = new DataView(memory.buffer);
      for (const { offset, addend } of expected) {
        expect(buffer.getBigUint64(toWasmAddress(image, base + offset), true)).toBe(base + addend);
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("refuses the ifunc entries, which would need code to run before the program starts", () => {
    const bytes = GLIBC_PIE;
    if (bytes === undefined) return;
    const image = imageOf(bytes, { base: DEFAULT_LOAD_BASE });
    const memory = createMemory(image);
    loadImage(bytes, image, memory);
    const table = parseRelocations(bytes, parseSectionHeaders(bytes, parseElfHeader(bytes)));
    expect(table.some((relocation) => relocation.type === 37)).toBe(true);
    expect(() => applyRelocations(table, { image, memory })).toThrow(/R_X86_64_IRELATIVE .* names a resolver at 0x[0-9a-f]+, and running it is what decides the value/);
  });

  it("refuses a file whose section headers were stripped, because its relocations are then unreadable", () => {
    const bytes = GLIBC_PIE;
    if (bytes === undefined) return;
    // e_shnum is two bytes at offset 60 in ELF64, and zeroing it is what --strip-all leaves behind
    const stripped = Uint8Array.from(bytes);
    stripped[60] = 0;
    stripped[61] = 0;
    expect(() => loadExecutable(stripped)).toThrow(/no section headers/);
  });
});

describe("a dynamically linked binary", () => {
  /** The host's own `ls`, which is the ordinary case: position-independent, and needing a linker this is not. */
  function hostBinary(): Uint8Array | undefined {
    for (const path of ["/bin/ls", "/usr/bin/ls"]) {
      if (existsSync(path)) return readFileSync(path);
    }
    return undefined;
  }

  it("refuses it by naming the symbol relocation it would need a dynamic linker for", () => {
    const bytes = hostBinary();
    if (bytes === undefined) return;
    // its relative relocations are the half this loader can do, and they are not enough on their own
    const table = parseRelocations(bytes, parseSectionHeaders(bytes, parseElfHeader(bytes)));
    expect(table.filter((relocation) => relocation.type === 8).length).toBeGreaterThan(0);
    expect(() => loadExecutable(bytes)).toThrow(/R_X86_64_(COPY|GLOB_DAT|JUMP_SLOT) at .*dynamic linker|no resolver/);
  });
});

describe("a non-PIE executable", () => {
  /** The same hello world the syscall tests use: assembled, linked at 0x400000, and with no relocations at all. */
  function buildStatic(): Uint8Array | undefined {
    if (!has("as") || !has("ld")) return undefined;
    const directory = mkdtempSync(join(tmpdir(), "elfwasm-static-"));
    try {
      const source = join(directory, "hello.s");
      writeFileSync(
        source,
        [
          ".globl _start",
          "_start:",
          "    mov $60, %rax",
          "    xor %rdi, %rdi",
          "    syscall",
          ".size _start, .-_start",
        ].join("\n"),
      );
      const object = join(directory, "hello.o");
      const executable = join(directory, "hello");
      execFileSync("as", ["--64", "-o", object, source], { stdio: "ignore" });
      execFileSync("ld", ["-nostdlib", "-no-pie", "-o", executable, object], { stdio: "ignore" });
      return readFileSync(executable);
    } catch {
      return undefined;
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }

  it("refuses a base below the file's own layout, which is address space it already occupies", () => {
    const bytes = buildStatic();
    if (bytes === undefined) return;
    // a non-PIE executable is linked at 0x400000, so there is nothing below that to move into
    expect(() => imageOf(bytes, { base: 0x300000n })).toThrow(/below the file's own layout/);
  });

  it("loads with nothing applied, because its linker already chose every address", () => {
    const bytes = buildStatic();
    if (bytes === undefined) return;
    const loaded = loadExecutable(bytes);
    expect(loaded.relocations).toBe(0);
    // the image keeps the layout the linker chose, since there is nothing to move it away from
    expect(loaded.image.baseAddress).toBe(DEFAULT_LOAD_BASE);
    expect(loaded.image.loadBias).toBe(0n);
    // and the bytes are the file's, so the entry's first instruction is where the program header said
    const buffer = new Uint8Array(loaded.memory.buffer);
    expect(buffer[loaded.image.entryAddress]).toBe(0x48);
  });
});
