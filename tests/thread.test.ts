/**
 * The thread pointer: `arch_prctl` setting `%fs`'s base, and `%fs:` addressing reading through it.
 *
 * Every case here is a program Linux can run too, so the expectation is never this translator's opinion: the same bytes run
 * natively and translated, and the two exit statuses are compared. That matters more than usual for `%fs`, because the wrong
 * base still produces a plausible read: `%fs:0x28` read as address `0x28` would return four bytes of the ELF header and the
 * canary check would then pass for the wrong reason.
 */

import { describe, expect, it } from "vitest";

import { imageOf, loadImage } from "../src/image.js";
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

interface Agreement {
  readonly native: number;
  readonly translated: number;
}

/**
 * Assemble a program, run it on the processor, then translate and run it here, and report both exit statuses.
 *
 * The program is linked `-no-pie`, so its addresses are the ones the linker chose and the image needs no relocating: what is
 * under test is `%fs`, not the loader.
 */
function agree(source: string): Agreement {
  const directory = mkdtempSync(join(tmpdir(), "elfwasm-thread-"));
  try {
    const path = join(directory, "program.s");
    writeFileSync(path, source);
    const object = join(directory, "program.o");
    const executable = join(directory, "program");
    execFileSync("as", ["--64", "-o", object, path], { stdio: "ignore" });
    execFileSync("ld", ["-nostdlib", "-no-pie", "-o", executable, object], { stdio: "ignore" });

    let native = 0;
    try {
      execFileSync(executable, [], { stdio: "ignore" });
    } catch (error) {
      native = (error as { status?: number }).status ?? -1;
    }

    const bytes = readFileSync(executable);
    const known = imageOf(bytes);
    const code = symbol(executable, "_start");
    const machine = new Machine({ image: known, pages: known.pages });
    loadImage(bytes, known, machine.memory);
    const offset = Number(code.address - known.baseAddress);
    const module = translateProgram(bytes.subarray(offset, offset + code.size), "main", {
      addresses: machine.addresses,
      codeAddress: code.address,
      code: { bytes, virtualAddress: known.baseAddress },
      memoryPages: known.pages,
      syscalls: true,
    });
    return { native, translated: machine.run(module).exitCode };
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

/**
 * Set `%fs` to `block`, then carry on: the four instructions every case starts with.
 *
 * `arch_prctl` is called with the block's own address, which is what a libc's thread setup does with its TLS block, and the
 * return value is not checked here because every case below would fail loudly if the call had not worked.
 */
function pointFsAt(block: string): string {
  return `    mov $158, %rax          # arch_prctl
    mov $0x1002, %rdi       # ARCH_SET_FS
    lea ${block}(%rip), %rsi
    syscall
`;
}

const FAIL = `    mov $99, %rdi
    mov $60, %rax
    syscall
`;

describe("the thread pointer, against Linux", () => {
  it.skipIf(!toolchain)(
    "writes and reads a canary through %fs: at the base it was pointed at",
    () => {
      const outcome = agree(`.globl _start
_start:
${pointFsAt("tls")}
    cmpq $0, %rax           # the call itself must have succeeded
    jne failed
    movq $42, %fs:0x28      # where a canary lives
    mov %fs:0x28, %rdi
    mov $60, %rax
    syscall
failed:
${FAIL}.size _start, .-_start
.data
.align 16
tls:
    .zero 256
`);
      expect(outcome.native).toBe(42);
      expect(outcome.translated).toBe(42);
    },
  );

  it.skipIf(!toolchain)(
    "reads the base at run time, so pointing %fs elsewhere changes what the same instruction reads",
    () => {
      /*
       * Five, nine, then back to the first block. A translation that resolved the base once would still be pointing at the
       * first block for the third write, and would read nine.
       */
      const outcome = agree(`.globl _start
_start:
${pointFsAt("tls1")}
    movq $5, %fs:0
${pointFsAt("tls2")}
    movq $9, %fs:0
${pointFsAt("tls1")}
    mov %fs:0, %rdi
    mov $60, %rax
    syscall
.size _start, .-_start
.data
.align 16
tls1:
    .zero 128
tls2:
    .zero 128
`);
      expect(outcome.native).toBe(5);
      expect(outcome.translated).toBe(5);
    },
  );

  it.skipIf(!toolchain)(
    "treats a base register as an offset inside the block, not as an address",
    () => {
      // %fs:(%rcx) is the thread pointer plus %rcx, which is how a thread-local array is indexed
      const outcome = agree(`.globl _start
_start:
${pointFsAt("tls")}
    mov $0x40, %rcx
    movq $7, %fs:(%rcx)
    mov %fs:0x40, %rdi
    mov $60, %rax
    syscall
.size _start, .-_start
.data
.align 16
tls:
    .zero 256
`);
      expect(outcome.native).toBe(7);
      expect(outcome.translated).toBe(7);
    },
  );

  it.skipIf(!toolchain)(
    "reports the base back through ARCH_GET_FS",
    () => {
      // a program that inherited its thread pointer has no other way to find it again, so the round trip is the contract
      const outcome = agree(`.globl _start
_start:
${pointFsAt("tls")}
    mov $158, %rax
    mov $0x1003, %rdi       # ARCH_GET_FS
    lea saved(%rip), %rsi
    syscall
    cmpq $0, %rax
    jne failed
    mov saved(%rip), %rdi
    lea tls(%rip), %rax
    cmp %rax, %rdi
    jne failed
    xor %rdi, %rdi
    mov $60, %rax
    syscall
failed:
${FAIL}.size _start, .-_start
.data
.align 16
saved:
    .quad 0
tls:
    .zero 128
`);
      expect(outcome.native).toBe(0);
      expect(outcome.translated).toBe(0);
    },
  );

  it.skipIf(!toolchain)(
    "answers -EINVAL for an operation it does not have and -EFAULT for an address it cannot write",
    () => {
      const outcome = agree(`.globl _start
_start:
    mov $158, %rax
    mov $0x9999, %rdi       # not an operation this kernel has
    xor %rsi, %rsi
    syscall
    cmp $-22, %rax          # -EINVAL
    jne failed
    mov $158, %rax
    mov $0x1003, %rdi       # ARCH_GET_FS
    xor %rsi, %rsi          # address 0, which is not in the image
    syscall
    cmp $-14, %rax          # -EFAULT
    jne failed
    xor %rdi, %rdi
    mov $60, %rax
    syscall
failed:
${FAIL}.size _start, .-_start
`);
      expect(outcome.native).toBe(0);
      expect(outcome.translated).toBe(0);
    },
  );

  it("refuses `lea` from a %fs: operand, because the processor ignores the override there", () => {
    /*
     * `lea %fs:0x10, %rax` loads 0x10 on a real processor: the assembler warns that the override is ineffectual, and the run
     * above confirmed it. Translating it as the thread pointer plus sixteen would be a translator that disagrees with the
     * hardware, so it is refused and the reason is in the message.
     */
    const machine = new Machine({ image: imageOf(BYTES), pages: 1 });
    expect(() =>
      translateProgram(
        // lea %fs:0x10, %rax ; ret
        Uint8Array.from([0x64, 0x48, 0x8d, 0x04, 0x25, 0x10, 0x00, 0x00, 0x00, 0xc3]),
        "main",
        {
          addresses: machine.addresses,
          codeAddress: 0x400800n,
          code: { bytes: BYTES, virtualAddress: 0x400000n },
          memoryPages: 1,
        },
      ),
    ).toThrow(/the processor itself ignores the override/);
  });
});

describe("a canary check written by gcc", () => {
  /**
   * A function compiled with `-fstack-protector-strong`, whose prologue reads the canary from `%fs:0x28` and whose epilogue
   * `sub`s it back out and branches if the two differ.
   *
   * This is the shape `%fs` exists for, and it comes from a compiler rather than from a test author: `mov %fs:0x28,%rax`,
   * `mov %rax,0x18(%rsp)`, then `sub %fs:0x28,%rdx ; jne __stack_chk_fail`. The stack is real here, so the entry prologue sets
   * `%rsp` and the frame is the one gcc asked for.
   */
  const C_SOURCE = `__attribute__((noinline)) int bump(int x) {
    volatile int slots[4];
    slots[0] = x;
    slots[1] = x + 1;
    return slots[0] + slots[1];
}
`;

  const ASM_SOURCE = `.globl _start
_start:
    mov $158, %rax          # arch_prctl
    mov $0x1002, %rdi       # ARCH_SET_FS
    lea tls(%rip), %rsi
    syscall
    mov $5, %rdi
    call bump
    mov %rax, %rdi
    mov $60, %rax
    syscall
.globl __stack_chk_fail
__stack_chk_fail:
    mov $99, %rdi
    mov $60, %rax
    syscall
.size _start, .-_start
.data
.align 16
tls:
    .zero 256
`;

  it.skipIf(!toolchain || !has("gcc"))(
    "returns the same value as the processor does, with the canary intact",
    () => {
      const directory = mkdtempSync(join(tmpdir(), "elfwasm-canary-"));
      try {
        const source = join(directory, "bump.c");
        const assembly = join(directory, "start.s");
        writeFileSync(source, C_SOURCE);
        writeFileSync(assembly, ASM_SOURCE);
        const object = join(directory, "bump.o");
        const start = join(directory, "start.o");
        const executable = join(directory, "program");
        execFileSync(
          "gcc",
          ["-c", "-O1", "-fstack-protector-strong", "-fno-pic", "-fno-asynchronous-unwind-tables", "-o", object, source],
          { stdio: "ignore" },
        );
        execFileSync("as", ["--64", "-o", start, assembly], { stdio: "ignore" });
        execFileSync("ld", ["-nostdlib", "-no-pie", "-o", executable, start, object], { stdio: "ignore" });

        let native = 0;
        try {
          execFileSync(executable, [], { stdio: "ignore" });
        } catch (error) {
          native = (error as { status?: number }).status ?? -1;
        }

        const bytes = readFileSync(executable);
        const known = imageOf(bytes);
        const code = symbol(executable, "_start");
        const machine = new Machine({ image: known, pages: known.pages + 1 });
        loadImage(bytes, known, machine.memory);
        const offset = Number(code.address - known.baseAddress);
        const module = translateProgram(bytes.subarray(offset, offset + code.size), "main", {
          addresses: machine.addresses,
          codeAddress: code.address,
          code: { bytes, virtualAddress: known.baseAddress },
          memoryPages: known.pages + 1,
          stackPages: 1,
          syscalls: true,
        });
        // five and six through a real frame, with the canary read and compared on the way out
        expect(native).toBe(11);
        expect(machine.run(module).exitCode).toBe(native);
      } finally {
        rmSync(directory, { recursive: true, force: true });
      }
    },
  );
});

describe("a %fs: access with no thread pointer", () => {
  it("traps instead of reading the image at the offset", () => {
    /*
     * No `arch_prctl`, so the base is zero, and the bound check is the thing that saves this: WASM memory starts at the image's
     * base, so address 0 is outside it and an access there traps rather than reading whatever the file's first bytes are.
     */
    const machine = new Machine({ image: imageOf(BYTES), pages: 1 });
    const module = translateProgram(
      // mov %fs:0x28, %rax ; ret
      Uint8Array.from([0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00, 0xc3]),
      "main",
      {
        addresses: machine.addresses,
        codeAddress: 0x400000n,
        code: { bytes: BYTES, virtualAddress: 0x400000n },
        memoryPages: 1,
      },
    );
    expect(() => machine.run(module)).toThrow();
  });
});

/** A minimal non-PIE image at 0x400000, so the translated code above has something to address. */
const BYTES = ((): Uint8Array => {
  const bytes = new Uint8Array(0x1000);
  const view = new DataView(bytes.buffer);
  // ELF64 header: little-endian, executable, x86-64, one program header
  bytes.set([0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00], 0);
  view.setUint16(16, 2, true); // e_type = ET_EXEC
  view.setUint16(18, 0x3e, true); // e_machine = x86-64
  view.setUint32(20, 1, true); // e_version
  view.setBigUint64(24, 0x400800n, true); // e_entry, inside the segment below
  view.setBigUint64(32, 64n, true); // e_phoff
  view.setUint16(52, 64, true); // e_ehsize
  view.setUint16(54, 56, true); // e_phentsize
  view.setUint16(56, 1, true); // e_phnum
  // one PT_LOAD covering the whole file at 0x400000, readable, writable and executable
  view.setUint32(64, 1, true); // p_type = PT_LOAD
  view.setUint32(64 + 4, 7, true); // p_flags = R|W|X
  view.setBigUint64(64 + 8, 0n, true); // p_offset
  view.setBigUint64(64 + 16, 0x400000n, true); // p_vaddr
  view.setBigUint64(64 + 24, 0x400000n, true); // p_paddr
  view.setBigUint64(64 + 32, 0x1000n, true); // p_filesz
  view.setBigUint64(64 + 40, 0x1000n, true); // p_memsz
  view.setBigUint64(64 + 48, 0x1000n, true); // p_align
  return bytes;
})();
