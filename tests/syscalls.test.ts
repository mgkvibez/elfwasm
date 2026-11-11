/**
 * The syscall layer: `write` and `exit_group`, answered by the machine rather than by the processor.
 *
 * Two kinds of check live here. The first is unit-level: a hand-written program that writes a known buffer and exits with a
 * known status, run against the shim. The second is the milestone this repository has been pointing at since the first
 * increment: a real `hello world`, assembled and linked, executed by Linux, and then translated and executed by WASM with its
 * stdout and its exit status compared against Linux's. Agreement there is what makes the word "translator" mean something.
 */

import { describe, expect, it } from "vitest";

// @ts-expect-error node:child_process is intentionally outside this project's type surface
const { execFileSync } = await import("node:child_process");
// @ts-expect-error node:fs is intentionally outside this project's type surface
const { mkdtempSync, readFileSync, rmSync, writeFileSync } = await import("node:fs");
// @ts-expect-error node:os is intentionally outside this project's type surface
const { tmpdir } = await import("node:os");
// @ts-expect-error node:path is intentionally outside this project's type surface
const { join } = await import("node:path");

import { Machine } from "../src/syscalls.js";
import { translateProgram } from "../src/translate.js";
import { imageOf, loadImage } from "../src/image.js";
import type { MemoryImage } from "../src/image.js";

const ORIGIN = 0x400000n;
const PAGE = 0x10000n;

/** One page of memory at the origin, which is where both the code and the buffer a test writes from live. */
const KNOWN: MemoryImage = {
  baseAddress: ORIGIN,
  loadBias: 0n,
  wasmBytes: Number(PAGE),
  pages: 1,
  entryAddress: 0,
  segments: [
    {
      virtualAddress: ORIGIN,
      fileOffset: 0n,
      wasmAddress: 0,
      // one segment covering the whole page, so any address inside it maps and anything outside it does not
      fileSize: PAGE,
      memorySize: PAGE,
      permissions: { read: true, write: true, execute: true },
    },
  ],
};

function hex(text: string): Uint8Array {
  return Uint8Array.from(text.trim().split(/\s+/).map((byte) => Number.parseInt(byte, 16)));
}

/**
 * A program that writes `text` from 0x400200 to stdout and exits with `status`.
 *
 * mov $0x1, %rax (write) ; mov $0x1, %rdi ; mov $0x400200, %rsi ; mov $len, %rdx ; syscall
 * mov $0x3c, %rax (exit) ; mov $status, %rdi ; syscall
 */
function writer(text: string, status = 0): Uint8Array {
  const at = (address: number): string =>
    [address & 0xff, (address >> 8) & 0xff, (address >> 16) & 0xff, (address >> 24) & 0xff]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join(" ");
  return hex(
    [
      "48 c7 c0 01 00 00 00", // mov $1, %rax
      "48 c7 c7 01 00 00 00", // mov $1, %rdi
      `48 c7 c6 ${at(0x400200)}`, // mov $0x400200, %rsi
      `48 c7 c2 ${at(text.length)}`, // mov $len, %rdx
      "0f 05", // syscall
      "48 c7 c0 3c 00 00 00", // mov $60, %rax
      `48 c7 c7 ${at(status)}`, // mov $status, %rdi
      "0f 05", // syscall
      "c3",
    ].join(" "),
  );
}

/** Translates and runs, copying `payload` in at 0x400200, which is what a loader does for a `.data` segment. */
function run(bytes: Uint8Array, payload = "", pages = 1) {
  const machine = new Machine({ image: KNOWN, pages });
  const module = translateProgram(bytes, "main", {
    addresses: machine.addresses,
    codeAddress: ORIGIN,
    memoryPages: pages,
    syscalls: true,
  });
  new Uint8Array(machine.memory.buffer).set(Uint8Array.from(payload, (c) => c.charCodeAt(0)), 0x200);
  return machine.run(module);
}

/** The WASM surface a test needs, declared here because this project compiles without ambient types. */
const runtime = (
  globalThis as unknown as {
    WebAssembly: {
      Module: new (bytes: Uint8Array) => unknown;
      Instance: new (module: unknown, imports?: unknown) => { readonly exports: Record<string, unknown> };
    };
  }
).WebAssembly;

describe("write and exit", () => {
  it("writes a buffer to stdout and exits with a status", () => {
    const outcome = run(writer("hello\n", 3), "hello\n");
    expect(outcome.stdout).toEqual(Uint8Array.from([...("hello\n" as string)].map((c) => c.charCodeAt(0))));
    expect(outcome.exitCode).toBe(3);
  });

  it("reports an exit without treating it as a failure", () => {
    // exit_group (231) rather than exit (60), which is what a libc's exit calls
    const bytes = hex("48 c7 c0 e7 00 00 00 48 c7 c7 02 00 00 00 0f 05 c3");
    expect(run(bytes).exitCode).toBe(2);
  });

  it("returns ENOSYS for a call the layer does not implement", () => {
    // getpid (39), whose number comes back in %rax as -38
    const bytes = hex("48 c7 c0 27 00 00 00 0f 05 c3");
    const machine = new Machine({ image: KNOWN, pages: 1 });
    const module = translateProgram(bytes, "main", {
      addresses: machine.addresses,
      codeAddress: ORIGIN,
      memoryPages: 1,
      syscalls: true,
    });
    const instance = new runtime.Instance(new runtime.Module(module), machine.imports);
    // the exported function's result is %rax, which is where a syscall leaves its answer
    const result = (instance.exports.main as () => bigint)();
    expect(result).toBe(-38n);
  });

  it("refuses a write whose buffer is outside the image, rather than reading memory", () => {
    // the buffer points past the end of the one page the machine has
    const bytes = hex(
      "48 c7 c0 01 00 00 00 48 c7 c7 01 00 00 00 48 c7 c6 00 00 50 00 48 c7 c2 04 00 00 00 0f 05 c3",
    );
    const machine = new Machine({ image: KNOWN, pages: 1 });
    const module = translateProgram(bytes, "main", {
      addresses: machine.addresses,
      codeAddress: ORIGIN,
      memoryPages: 1,
      syscalls: true,
    });
    const instance = new runtime.Instance(new runtime.Module(module), machine.imports);
    expect((instance.exports.main as () => bigint)()).toBe(-14n);
  });
});

describe("the import, and what it does to every function index", () => {
  /**
   * entry at 0x400000:  call 0x400020 ; mov %rax, %rdi ; mov $60, %rax ; syscall
   * callee at 0x400020: mov $7, %rax ; ret
   *
   * An imported function occupies index 0, so the entry is function 1 and the callee function 2. A `call` that kept the
   * pre-import index would call the entry again, which recurses until the engine gives up; a `call` that skipped a function
   * would fail validation. The callee's 7 becomes the exit status, so the status is the evidence it was reached.
   */
  const CODE = hex(
    "e8 1b 00 00 00 48 89 c7 48 c7 c0 3c 00 00 00 0f 05 " + // call ; mov %rax,%rdi ; mov $60,%rax ; syscall
      "90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 " +
      "48 c7 c0 07 00 00 00 c3", // mov $7, %rax ; ret
  );

  it("still reaches a callee with the import in front of it", () => {
    const machine = new Machine({ image: KNOWN, pages: 1 });
    const module = translateProgram(CODE.subarray(0, 0x11), "main", {
      addresses: machine.addresses,
      codeAddress: ORIGIN,
      code: { bytes: CODE, virtualAddress: ORIGIN },
      memoryPages: 1,
      syscalls: true,
    });
    expect(machine.run(module).exitCode).toBe(7);
  });
});

describe("hello world, against Linux", () => {
  const has = (binary: string): boolean => {
    try {
      execFileSync("sh", ["-c", `command -v ${binary}`], { stdio: "ignore" });
      return true;
    } catch {
      return false;
    }
  };

  const symbol = (executable: string, name: string): { address: bigint; size: number } => {
    const listed = execFileSync("nm", ["-S", executable], { encoding: "utf8" }) as string;
    for (const line of listed.split("\n")) {
      const fields = line.trim().split(/\s+/);
      if (fields.length >= 4 && fields[3] === name) {
        return { address: BigInt(`0x${fields[0]!}`), size: Number.parseInt(fields[1]!, 16) };
      }
    }
    throw new Error(`no sized symbol named ${name} in ${executable}`);
  };

  it.skipIf(!has("as") || !has("ld") || !has("nm"))(
    "produces the same stdout and exit status as the processor does",
    () => {
      const directory = mkdtempSync(join(tmpdir(), "elfwasm-hello-"));
      try {
        const source = join(directory, "hello.s");
        writeFileSync(
          source,
          `.globl _start
_start:
    mov $1, %rax          # write
    mov $1, %rdi          # stdout
    lea message(%rip), %rsi
    mov $14, %rdx
    syscall
    mov $60, %rax         # exit
    xor %rdi, %rdi
    syscall
.size _start, .-_start
message:
    .ascii "Hello, world!\\n"
`,
        );
        const executable = join(directory, "hello");
        execFileSync("as", ["--64", "-o", join(directory, "hello.o"), source]);
        execFileSync("ld", ["-nostdlib", "-no-pie", "-o", executable, join(directory, "hello.o")]);

        // Linux's answer: the bytes on stdout, and the status
        let nativeExit = 0;
        let nativeOut = "";
        try {
          nativeOut = execFileSync(executable, [], { encoding: "utf8" });
        } catch (error) {
          nativeExit = (error as { status?: number }).status ?? 0;
          nativeOut = String((error as { stdout?: string }).stdout ?? "");
        }
        expect(nativeOut).toBe("Hello, world!\n");
        expect(nativeExit).toBe(0);

        /*
         * The translated answer. The image is built from the executable's own program headers, so the addresses in the code are
         * the addresses the loader would have given it, and `lea message(%rip)` resolves against them.
         */
        const bytes = readFileSync(executable);
        const known = imageOf(bytes);
        const code = symbol(executable, "_start");
        const machine = new Machine({ image: known, pages: known.pages });
        loadImage(bytes, known, machine.memory);

        // the entry function is `_start`'s own bytes, and a call inside it is followed into the whole file
        const offset = Number(code.address - known.baseAddress);
        const module = translateProgram(bytes.subarray(offset, offset + code.size), "main", {
          addresses: machine.addresses,
          codeAddress: code.address,
          code: { bytes, virtualAddress: known.baseAddress },
          memoryPages: known.pages,
          syscalls: true,
        });
        const outcome = machine.run(module);
        expect(String.fromCharCode(...outcome.stdout)).toBe(nativeOut);
        expect(outcome.exitCode).toBe(nativeExit);
        // the message really was read out of the loaded image, not out of the code bytes
        expect(String.fromCharCode(...outcome.stdout)).toContain("Hello, world!");
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    },
  );
});
