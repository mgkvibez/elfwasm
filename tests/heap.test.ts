/**
 * `brk` and `mmap`: the two ways a program gets memory that is not in its file.
 *
 * The model is deliberately small and stated rather than implied. The heap begins where the image's own pages end, so a virtual
 * address and a WASM offset stay one addition apart; `brk` grows upward from there; `mmap` grows downward from the top of the
 * free region, which is the direction Linux grows its own mappings; and neither may cross the other. A page below the stack is a
 * page the program may have, and the stack is whatever the caller said it was.
 */

import { describe, expect, it } from "vitest";

import type { MemoryImage } from "../src/image.js";
import { imageOf, loadImage } from "../src/image.js";
import { ERRNO, Machine, SYSCALLS } from "../src/syscalls.js";
import { translateProgram } from "../src/translate.js";

// @ts-expect-error node:child_process is intentionally outside this project's type surface
const { execFileSync } = await import("node:child_process");
// @ts-expect-error node:fs is intentionally outside this project's type surface
const { mkdtempSync, readFileSync, rmSync, writeFileSync } = await import("node:fs");
// @ts-expect-error node:os is intentionally outside this project's type surface
const { tmpdir } = await import("node:os");
// @ts-expect-error node:path is intentionally outside this project's type surface
const { join } = await import("node:path");

const ORIGIN = 0x400000n;
const WASM_PAGE = 0x10000n;

/** One page of image at the origin, so the heap starts at `ORIGIN + 0x10000` and everything above it is free. */
const KNOWN: MemoryImage = {
  baseAddress: ORIGIN,
  loadBias: 0n,
  wasmBytes: Number(WASM_PAGE),
  pages: 1,
  entryAddress: 0,
  segments: [
    {
      virtualAddress: ORIGIN,
      fileOffset: 0n,
      wasmAddress: 0,
      fileSize: WASM_PAGE,
      memorySize: WASM_PAGE,
      permissions: { read: true, write: true, execute: true },
    },
  ],
};

/** Calls the imported kernel the way a translated `syscall` does, with six arguments and the trailing ones zeroed. */
function kernel(machine: Machine): (number: bigint, ...args: bigint[]) => bigint {
  const syscall = machine.imports.env.syscall as (...values: bigint[]) => bigint;
  return (number: bigint, ...args: bigint[]) =>
    syscall(number, ...args, ...Array.from({ length: 6 - args.length }, () => 0n));
}

const MAP_PRIVATE = 0x02n;
const MAP_FIXED = 0x10n;
const MAP_ANONYMOUS = 0x20n;
const PAGE = 0x1000n;

describe("brk", () => {
  const machine = (): Machine => new Machine({ image: KNOWN, pages: 4 });

  it("starts where the image's own pages end, and reports it for a break of zero", () => {
    const call = kernel(machine());
    expect(call(SYSCALLS.brk, 0n)).toBe(ORIGIN + WASM_PAGE);
  });

  it("moves the break and makes the new memory readable and writable", () => {
    const subject = machine();
    const call = kernel(subject);
    const start = call(SYSCALLS.brk, 0n);
    expect(call(SYSCALLS.brk, start + PAGE)).toBe(start + PAGE);

    const bytes = subject.bytesAt(start, 4);
    expect(bytes).toBeDefined();
    bytes!.set([0x41, 0x42, 0x43, 0x44]);
    // the same bytes, read back through the kernel's own view of the memory a `write` would copy from
    expect([...subject.bytesAt(start, 4)!]).toEqual([0x41, 0x42, 0x43, 0x44]);
  });

  it("refuses to grow past the memory, answering with the old break", () => {
    const call = kernel(machine());
    const start = call(SYSCALLS.brk, 0n);
    // four pages in total, one of them the image, so the limit is the top of the memory
    const limit = ORIGIN + 4n * WASM_PAGE;
    expect(call(SYSCALLS.brk, limit)).toBe(limit);
    expect(call(SYSCALLS.brk, limit + 1n)).toBe(limit);
    expect(call(SYSCALLS.brk, 0n)).toBe(limit);
    expect(start).toBeLessThan(limit);
  });

  it("refuses a break below the image, and one that would cross into the stack", () => {
    const subject = new Machine({ image: KNOWN, pages: 4, stackPages: 1 });
    const call = kernel(subject);
    expect(call(SYSCALLS.brk, ORIGIN)).toBe(ORIGIN + WASM_PAGE);
    const limit = ORIGIN + 3n * WASM_PAGE;
    expect(call(SYSCALLS.brk, limit)).toBe(limit);
    // the top page is the stack's, so one byte more is a byte of somebody's frame
    expect(call(SYSCALLS.brk, limit + 1n)).toBe(limit);
  });

  it("zeroes what it hands out, including a range that was written before it was given back", () => {
    const subject = machine();
    const call = kernel(subject);
    const start = call(SYSCALLS.brk, 0n);
    expect(call(SYSCALLS.brk, start + PAGE)).toBe(start + PAGE);
    subject.bytesAt(start, 1)!.set([0xaa]);
    // shrink, then grow again: the page had a byte in it, and a fresh page does not
    call(SYSCALLS.brk, start);
    expect(call(SYSCALLS.brk, start + PAGE)).toBe(start + PAGE);
    expect(subject.bytesAt(start, 1)![0]).toBe(0);
  });
});

describe("mmap", () => {
  const machine = (): Machine => new Machine({ image: KNOWN, pages: 4 });

  it("hands out a page-aligned anonymous mapping below the top of the free region", () => {
    const subject = machine();
    const call = kernel(subject);
    const mapped = call(SYSCALLS.mmap, 0n, PAGE, 3n, MAP_PRIVATE | MAP_ANONYMOUS, -1n);
    expect(mapped).toBe(ORIGIN + 4n * WASM_PAGE - PAGE);
    expect(mapped % PAGE).toBe(0n);
    expect([...subject.bytesAt(mapped, 4)!]).toEqual([0, 0, 0, 0]);
    // and it is writable, which is the point of asking for it
    subject.bytesAt(mapped, 2)!.set([1, 2]);
    expect([...subject.bytesAt(mapped, 2)!]).toEqual([1, 2]);
  });

  it("rounds the length up to a page, so two mappings do not overlap", () => {
    const subject = machine();
    const call = kernel(subject);
    const first = call(SYSCALLS.mmap, 0n, 1n, 3n, MAP_PRIVATE | MAP_ANONYMOUS, -1n);
    const second = call(SYSCALLS.mmap, 0n, 1n, 3n, MAP_PRIVATE | MAP_ANONYMOUS, -1n);
    // one byte is one page, and the second mapping is a whole page below the first
    expect(first - second).toBe(PAGE);
  });

  it("refuses a file mapping and a fixed address rather than serving anonymous memory", () => {
    const call = kernel(machine());
    // a private mapping of a file: there is no file behind this address space to map
    expect(call(SYSCALLS.mmap, 0n, PAGE, 3n, MAP_PRIVATE, 3n)).toBe(ERRNO.ENOSYS);
    // MAP_FIXED would put the mapping wherever it was told, which may be the image
    expect(call(SYSCALLS.mmap, ORIGIN, PAGE, 3n, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1n)).toBe(ERRNO.ENOSYS);
  });

  it("refuses a zero length as an invalid argument", () => {
    expect(kernel(machine())(SYSCALLS.mmap, 0n, 0n, 3n, MAP_PRIVATE | MAP_ANONYMOUS, -1n)).toBe(ERRNO.EINVAL);
  });

  it("runs out of memory when the mappings reach the break, and then gives nothing away", () => {
    const subject = machine();
    const call = kernel(subject);
    const free = 3n * WASM_PAGE;
    expect(call(SYSCALLS.mmap, 0n, free, 3n, MAP_PRIVATE | MAP_ANONYMOUS, -1n)).toBe(ORIGIN + WASM_PAGE);
    // the heap is now empty, so a page more has nowhere to go
    expect(call(SYSCALLS.mmap, 0n, PAGE, 3n, MAP_PRIVATE | MAP_ANONYMOUS, -1n)).toBe(ERRNO.ENOMEM);
    // and the break cannot cross into what was mapped either
    expect(call(SYSCALLS.brk, ORIGIN + 2n * WASM_PAGE)).toBe(ORIGIN + WASM_PAGE);
  });
});

const has = (binary: string): boolean => {
  try {
    execFileSync("sh", ["-c", `command -v ${binary}`], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
};

const toolchain = has("as") && has("ld") && has("nm");

function symbol(executable: string, name: string): { address: bigint; size: number } {
  const listed = execFileSync("nm", ["-S", executable], { encoding: "utf8" }) as string;
  for (const line of listed.split("\n")) {
    const fields = line.trim().split(/\s+/);
    if (fields.length >= 4 && fields[3] === name) {
      return { address: BigInt(`0x${fields[0]!}`), size: Number.parseInt(fields[1]!, 16) };
    }
  }
  throw new Error(`no sized symbol named ${name} in ${executable}`);
}

describe("allocating, against Linux", () => {
  /**
   * Asks `brk` for a page, copies eight bytes off the image into it, and writes six of them to stdout.
   *
   * This is allocation end to end: the address comes back from the kernel, the program writes through it, and `write` reads it
   * back out of the same memory. If the break were wrong, the copy would land somewhere else; if `write` only looked inside the
   * image, it would answer `-EFAULT` where Linux answered with bytes.
   */
  const SOURCE = `.globl _start
_start:
    mov $12, %rax           # brk
    xor %rdi, %rdi
    syscall                 # %rax = where the break is now
    mov %rax, %r12
    lea 0x1000(%rax), %rdi
    mov $12, %rax
    syscall                 # grow it by a page; %rax = the new break
    cmp %r12, %rax
    jbe failed              # it has to have moved
    lea message(%rip), %rax
    mov (%rax), %rbx        # eight bytes, which is one whole message with its terminator
    mov %rbx, (%r12)        # into the page the kernel just handed over
    mov $1, %rax            # write
    mov $1, %rdi            # stdout
    mov %r12, %rsi          # from the heap, not from the image
    mov $6, %rdx
    syscall
    xor %rdi, %rdi
    mov $60, %rax
    syscall
failed:
    mov $99, %rdi
    mov $60, %rax
    syscall
.size _start, .-_start
.data
message:
    .ascii "heap!\\n\\0\\0"
`;

  it.skipIf(!toolchain)("prints what it allocated, with the same bytes and status as the processor", () => {
    const directory = mkdtempSync(join(tmpdir(), "elfwasm-heap-"));
    try {
      const source = join(directory, "heap.s");
      writeFileSync(source, SOURCE);
      const object = join(directory, "heap.o");
      const executable = join(directory, "heap");
      execFileSync("as", ["--64", "-o", object, source], { stdio: "ignore" });
      execFileSync("ld", ["-nostdlib", "-no-pie", "-o", executable, object], { stdio: "ignore" });

      // Linux first: the bytes it printed and the status it exited with
      let nativeExit = 0;
      let nativeOut = "";
      try {
        nativeOut = execFileSync(executable, [], { encoding: "utf8" }) as string;
      } catch (error) {
        nativeExit = (error as { status?: number }).status ?? -1;
        nativeOut = String((error as { stdout?: string }).stdout ?? "");
      }
      expect(nativeOut).toBe("heap!\n");
      expect(nativeExit).toBe(0);

      const bytes = readFileSync(executable);
      const known = imageOf(bytes);
      const code = symbol(executable, "_start");
      // four pages of heap above the image, which is what the program asks the kernel for
      const machine = new Machine({ image: known, pages: known.pages + 4 });
      loadImage(bytes, known, machine.memory);
      const offset = Number(code.address - known.baseAddress);
      const module = translateProgram(bytes.subarray(offset, offset + code.size), "main", {
        addresses: machine.addresses,
        codeAddress: code.address,
        code: { bytes, virtualAddress: known.baseAddress },
        memoryPages: known.pages + 4,
        syscalls: true,
      });
      const outcome = machine.run(module);
      expect(String.fromCharCode(...outcome.stdout)).toBe(nativeOut);
      expect(outcome.exitCode).toBe(nativeExit);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
