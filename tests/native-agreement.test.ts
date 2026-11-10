/**
 * Does the translator agree with the processor?
 *
 * Every other check in this repository compares the translator against something it produced itself: hand-written bytes on
 * one side, an expected number on the other. This one does not. It assembles real assembly with binutils, takes the exact
 * bytes `objcopy` extracts from the object file, and runs them twice: once by the machine, through a linked `_start` that
 * exits with the function's own result, and once through the decoder and translator as a WASM module. The two numbers, and
 * the bytes in between, have to agree.
 *
 * It skips rather than fails when binutils is not installed, because the suite's arrangement is that a cold checkout runs
 * with no toolchain at all. `tools/native/native.md` records how to run it.
 */

import { describe, expect, it } from "vitest";

// @ts-expect-error node:child_process is intentionally outside this project's type surface
const { execFileSync } = await import("node:child_process");
// @ts-expect-error node:fs is intentionally outside this project's type surface
const { mkdtempSync, readFileSync, readdirSync, rmSync } = await import("node:fs");
// @ts-expect-error node:os is intentionally outside this project's type surface
const { tmpdir } = await import("node:os");
// @ts-expect-error node:path is intentionally outside this project's type surface
const { join } = await import("node:path");

import { decode } from "../src/decode.js";
import { translate } from "../src/translate.js";

const SOURCE_DIRECTORY = "tools/native";

/** The entry point a linked program needs: call the function, then exit with its result as the status. */
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
      Module: new (bytes: Uint8Array) => unknown;
      Instance: new (module: unknown) => { exports: Record<string, () => bigint> };
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

/** Exit status, which is how a linked program returns its number here: a non-zero status is the expected case. */
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

function translatedResult(bytes: Uint8Array): number {
  const instance = new runtime.Instance(new runtime.Module(translate(decode(bytes))));
  return Number(instance.exports.main!());
}

const toolchain = ["as", "objcopy", "ld"].every(has);
const programs: string[] = (readdirSync(SOURCE_DIRECTORY) as string[])
  .filter((name) => name.endsWith(".s"))
  .sort();

describe.skipIf(!toolchain)("the translator against the processor, on assembler output", () => {
  it("agrees with native execution on every program, number for number", () => {
    const directory = mkdtempSync(join(tmpdir(), "elfwasm-native-"));
    const startPath = join(directory, "start.s");
    execFileSync("sh", ["-c", `cat > ${startPath} <<'EOF'\n${START}EOF`]);
    const results: string[] = [];
    try {
      for (const program of programs) {
        const objectFile = join(directory, `${program}.o`);
        const bytesFile = join(directory, `${program}.bin`);
        const runnerObject = join(directory, `${program}-start.o`);
        const executable = join(directory, program.replace(/\.s$/, ""));

        execFileSync("as", ["--64", "-o", objectFile, join(SOURCE_DIRECTORY, program)]);
        execFileSync("objcopy", ["-O", "binary", "--only-section=.text", objectFile, bytesFile]);
        execFileSync("as", ["--64", "-o", runnerObject, startPath]);
        // -nostdlib: no libc between the program and its own instructions
        execFileSync("ld", ["-nostdlib", "-o", executable, runnerObject, objectFile]);
        execFileSync("chmod", ["+x", executable]);

        const native = status(executable);
        const bytes = new Uint8Array(readFileSync(bytesFile));
        const translated = translatedResult(bytes);
        results.push(`${program}: native ${native}, translated ${translated}`);
        expect(translated & 0xff, results.join("; ")).toBe(native);
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
    // the tool delays reading the directory until collection time, so this is where the empty case surfaces
    expect(results.length).toBeGreaterThan(0);
  });
});
