/**
 * The translator: decoded instructions in, a WASM module out.
 *
 * Stage 3 is being built breadth first and honestly: one function, sixteen i64 locals for the sixteen registers, then
 * straight-line arithmetic, then flags and forward branches. Every instruction outside the covered set is refused by name
 * and offset, in the same spirit as the decoder, because a translator that silently skipped an instruction would return a
 * wrong number with no record of where it went wrong.
 *
 * Register layout: the sixteen 64-bit registers are `rax` to `r15` in encoding order. A 32-bit operand names the 64-bit
 * local and carries width 32, which is x86's own model: writing the low 32 bits zeroes the upper 32, and reading them
 * ignores it. The translation makes that explicit, because an i64 local that keeps its high bits through a 32-bit write is
 * the classic way to turn an `int` result into a different `long`.
 *
 * Flags are never materialised. `cmp` and `test` store their operands in two extra locals, and a `jcc` becomes the
 * comparison itself, at the width the operands were. That is why `cmp $-1, %eax ; jl` and `cmp $-1, %rax ; jl` differ
 * without either one being special-cased: the condition is evaluated at 32 bits or 64 bits, and x86's own rule that a
 * 32-bit compare ignores the upper half falls out of it.
 */

import {
  DecodeError,
  decode,
  decodeInstruction,
  type Instruction,
  type Operand,
  type RegisterName,
} from "./decode.js";
import {
  OP,
  globalGet,
  globalSet,
  i32Const,
  i64Const,
  localGet,
  localSet,
  memarg,
  module,
  unsigned,
} from "./emit.js";
import type { AddressSpace } from "./image.js";

/** The register file, in the order the encoding numbers it. Index is the WASM local. */
const REGISTERS = [
  "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
  /*
   * The sixteen vector registers, as two 64-bit halves each, which is what an `i64` global can hold. They come after the
   * general registers because the numbering of the general file is the encoding's, and the thread pointer is the global after
   * the last of these.
   *
   * A half is a register here rather than a pair of locals so that `movq %xmm0, 0x20(%rsp)` is the store it looks like: the
   * existing `mov`-to-memory path reads a register and writes an address, and the low half of a vector register is a register
   * that holds 64 bits.
   */
  ...Array.from({ length: 16 }, (_, number) => [`xmm${number}.lo`, `xmm${number}.hi`] as const).flat(),
] as const;

/** The two locals `cmp` and `test` leave their operands in, so a branch can be the comparison. */
/**
 * The scratch locals, which are the only locals left now that the registers are globals.
 *
 * A register has to outlive a call, so a register is a global. A comparison's two operands do not: they exist between the
 * instruction that sets them and the branch that reads them, both inside one function.
 */
const CMP_A = 0;
const CMP_B = 1;
/**
 * Where a computed address waits between its bound check and its use.
 *
 * It cannot be one of the comparison locals: `cmp (%rax), %rbx` keeps its first operand in `CMP_A` while the second is being
 * loaded, and an address check that wrote `CMP_A` there would silently compare the address against the value.
 */
const ADDR = 2;
/**
 * Where a vector instruction keeps the two halves it interleaves.
 *
 * Not `CMP_A` and `CMP_B`: those hold a comparison that a later branch reads, and an instruction that writes no flag must leave
 * them alone, so a `punpckldq` cannot borrow them any more than a byte store can.
 */
const SSE_A = 4;
const SSE_B = 5;
/**
 * Where a division keeps its long-division state.
 *
 * `%rdx:%rax` is a 128-bit dividend and WASM has no divide at that width, so `div` and `idiv` are a bit at a time: the remaining
 * remainder, the quotient being assembled, the bit a shift would drop, and whether the trial subtraction happened. `mul` borrows
 * the same slots for the four partial products a 64-by-64 widening multiply is made of.
 */
const DIV_DIVISOR = 6;
const DIV_LO = 7;
const DIV_HI = 8;
const DIV_REM = 9;
const DIV_Q = 10;
const DIV_CARRY = 11;
const DIV_TAKE = 12;
const DIV_SIGN_Q = 13;
const DIV_SIGN_HI = 14;
const DIV_FLAG = 15;
const DIV_FLAG2 = 16;
/**
 * Where an indirect call keeps the target it read.
 *
 * The target is compared against every function this program translated, so it has to be somewhere a comparison can read it more
 * than once, and a local is what a comparison reads.
 */
const INDIRECT = 17;
/** The carry a `sbb` subtracts, worked out from the comparison the flags still hold. */
const SBB_CARRY = 18;
/**
 * Where the block a body is executing is recorded, for a body whose control flow is not a tree.
 *
 * Registers are globals and a comparison's operands are locals, so a block of instructions that begins anywhere needs no state of
 * its own beyond which block to run next. This local is that.
 */
const PC = 3;
const LOCAL_COUNT = 19;

/** The imported syscall's function index, which is 0: an import comes before every function the module defines. */
const SYSCALL_INDEX = 0;

/** WASM's page size, which is what a bound is expressed in. */
const PAGE_SIZE = 64 * 1024;

/**
 * What the translator needs to know about the world outside the bytes.
 *
 * A memory operand names a virtual address, and turning that into a WASM offset needs the mapped image. Passing it in is
 * what keeps the translator a pure function of its input plus an explicit description of where the code will live, rather
 * than something that guesses at a base address.
 */
export interface TranslateOptions {
  /** The mapped image's address space. Without it, a memory operand is refused rather than assumed. */
  readonly addresses?: AddressSpace;
  /**
   * Reads the image's bytes at a virtual address, which is what a jump table is: data the walk has to read to know where a `switch`
   * goes. A file's bytes and its segments are enough to answer it, and the caller has both.
   */
  readonly readData?: (virtualAddress: bigint, width: number) => bigint | undefined;
  /** Virtual address of the first byte of `instructions`, for RIP-relative operands. Defaults to 0. */
  readonly codeAddress?: bigint;
  /** Pages to import when the body touches memory. Defaults to the image's own pages plus `stackPages`. */
  readonly memoryPages?: number;
  /**
   * The code the entry function was cut from, so a `call` can be followed to the function it names.
   *
   * Without it a `call` is refused rather than guessed at: the displacement is relative, and the target may well be a
   * function in a different image.
   */
  readonly code?: { readonly bytes: Uint8Array; readonly virtualAddress: bigint };
  /**
   * The ranges of the code region a processor would let a program execute, which is what a `PT_LOAD` segment with `PF_X` says.
   *
   * A call whose target is outside every one of them does not name a function: the bytes there are a header, a relocation table or
   * data, and jumping to one faults. A static PIE leaves exactly such a call in the path it never takes, `call 0`, whose target is
   * the image's own first byte. Translated as a call into a body that traps, the program keeps its own answer and the dead path
   * crashes where the machine would crash. Left out, the whole region counts as executable, which is what a caller that hands over
   * bytes and nothing about segments is saying.
   */
  readonly executable?: readonly { readonly start: bigint; readonly end: bigint }[];
  /**
   * Function addresses that nothing in the program calls directly, so that an indirect call can still be aimed at them.
   *
   * A function pointer handed to a program as an argument (`main`, whose address `_start` passes to `__libc_start_main`) never appears
   * as a call target in the bytes, and the translation of a call through a pointer would have nothing to compare it against.
   */
  readonly indirectTargets?: readonly bigint[];

  /**
   * Jumps in this body that leave its own bytes, by instruction offset, holding the address each one reaches.
   *
   * A `jmp` out of a body is a call followed by a return: what it reaches is a function that starts elsewhere, and its answer is this
   * function's answer. The body's instruction list cannot hold that target, because the target's bytes are not this body's, so the
   * walk records it here.
   */
  readonly tailCalls?: ReadonlyMap<number, bigint>;
  /**
   * Answer `syscall` through the imported host function, which is `Machine` in `src/syscalls.ts`.
   *
   * Off by default, like the stack, because it changes what the caller must supply: a module that can call the kernel needs an
   * import at instantiation, and a caller who leaves it out gets a refusal at translation rather than a link error at load.
   */
  readonly syscalls?: boolean;
  /** Filled in by `translate` itself. A caller never sets this. */
  readonly calls?: CallRegistry;
  /**
   * Virtual address of the first byte of the function being translated, which a `call` displacement is measured from.
   * Filled in per function by the program translation; a caller never sets it.
   */
  readonly functionAddress?: bigint;
  /**
   * Pages above the image reserved for the stack, and the opt-in that enables stack translation at all.
   *
   * It is not a default because it changes what the caller has to allocate: the import minimum grows, so a caller that
   * supplies the image's pages alone gets an instantiation error rather than a program with a stack it did not ask for.
   */
  readonly stackPages?: number;
}

/** The functions one translation knows about, in module order. Index 0 is the entry. */
export interface CallRegistry {
  /** The module index of the function at this virtual address, decoding it on first sight. */
  indexOf(address: bigint, instruction: Instruction): number;
  /** Every function translated so far, which is what an indirect call can be aimed at. */
  candidates?(): readonly bigint[];
}

export class TranslateError extends Error {
  readonly offset: number;

  constructor(offset: number, message: string) {
    /*
     * A negative offset is a real thing here: a branch that reaches a byte before the function's start is that far before it, and
     * printing it as `0x-100` would be neither the offset nor the address.
     */
    super(`at offset ${offset < 0 ? "-" : ""}0x${Math.abs(offset).toString(16)}: ${message}`);
    this.offset = offset;
  }
}

function localOfRegister(name: string, instruction: Instruction): number {
  // the 32-bit and 8-bit names share the 64-bit local, which is x86's own model: a narrow write lands in the same register
  const base = name.replace(/^r(\d+)[bwd]?$/, (_match, digits: string) => `r${digits}`);
  const index = REGISTERS.indexOf(base as (typeof REGISTERS)[number]);
  if (index < 0) {
    throw new TranslateError(instruction.offset, `register ${name} is not one of the 64-bit registers`);
  }
  return index;
}

/**
 * Where a narrow register sits inside its 64-bit local.
 *
 * A byte register is not a second variable, it is a byte of one: `%al` is bits 0 to 7 of `%rax`, and `%ah` is bits 8 to 15
 * of the same register. A write to either leaves the rest of `%rax` alone, which is only true if the write merges rather than
 * replaces. That is why this is a shift and a mask rather than a second local.
 */
interface RegisterSlot {
  readonly local: number;
  /** How far up the 64-bit local this register starts: 0 for `al`, 8 for `ah`. */
  readonly shift: number;
  /** The width of the register named, which is the width of the access, not of the local behind it. */
  readonly bits: 8 | 16 | 32 | 64;
}

/** The eight bit registers of the legacy registers, and the eight above `bl` that REX adds. */
const BYTE_TO_FULL: Readonly<Record<string, string>> = {
  al: "rax",
  cl: "rcx",
  dl: "rdx",
  bl: "rbx",
  spl: "rsp",
  bpl: "rbp",
  sil: "rsi",
  dil: "rdi",
  ...Object.fromEntries(Array.from({ length: 8 }, (_unused, index) => [`r${index + 8}b`, `r${index + 8}`])),
};

/** The sixteen 16-bit registers, which are the low half of a 64-bit register under a name of its own. */
const WORD_TO_FULL: Readonly<Record<string, string>> = {
  ax: "rax",
  cx: "rcx",
  dx: "rdx",
  bx: "rbx",
  sp: "rsp",
  bp: "rbp",
  si: "rsi",
  di: "rdi",
  ...Object.fromEntries(Array.from({ length: 8 }, (_unused, index) => [`r${index + 8}w`, `r${index + 8}`])),
};

/** `ah`, `ch`, `dh` and `bh`: the second byte of four registers, which no REX prefix can name. */
const HIGH_BYTES: Readonly<Record<string, string>> = { ah: "rax", ch: "rcx", dh: "rdx", bh: "rbx" };

function slotOf(operand: Operand, instruction: Instruction): RegisterSlot {
  if (operand.kind !== "reg") {
    throw new TranslateError(instruction.offset, `expected a register operand, found ${operand.kind}`);
  }
  const { name, bits } = operand;
  if (bits === 16) {
    const full = WORD_TO_FULL[name];
    if (full === undefined) {
      throw new TranslateError(instruction.offset, `${name} is not a 16-bit register this translator knows`);
    }
    return { local: localOfRegister(full, instruction), shift: 0, bits };
  }
  if (bits === 8) {
    const high = HIGH_BYTES[name];
    if (high !== undefined) return { local: localOfRegister(high, instruction), shift: 8, bits };
    const full = BYTE_TO_FULL[name];
    if (full === undefined) {
      throw new TranslateError(instruction.offset, `${name} is not an 8-bit register this translator knows`);
    }
    return { local: localOfRegister(full, instruction), shift: 0, bits };
  }
  return { local: localOfRegister(name, instruction), shift: 0, bits };
}

/** The eight bits a narrow register owns, as an i64 in the range a byte holds. */
function read(operand: Operand, instruction: Instruction): number[] {
  const slot = slotOf(operand, instruction);
  if (slot.bits === 32 || slot.bits === 64) return globalGet(slot.local);
  // a byte or a word: the same register seen through a mask, at the offset the name puts it
  const code = globalGet(slot.local);
  const placed = slot.shift === 0 ? code : [...code, ...i64Const(BigInt(slot.shift)), OP.i64ShrU];
  return [...placed, ...i64Const(slot.bits === 8 ? 0xffn : 0xffffn), OP.i64And];
}

/**
 * Writes a register, leaving the operand stack as it found it.
 *
 * A byte write cannot replace its local, because seven eighths of it belong to something else: `mov %al, %dh` must leave
 * `%rdx`'s other bytes alone, and `mov %dh, %al` must not disturb `%rax`'s. So the bits around the target are read back,
 * masked out, and put back with the new byte or'd in.
 */
function write(operand: Operand, instruction: Instruction): number[] {
  const slot = slotOf(operand, instruction);
  if (slot.bits === 32 || slot.bits === 64) return globalSet(slot.local);
  const owned = slot.bits === 8 ? 0xffn : 0xffffn;
  const placed = slot.shift === 0 ? [] : [...i64Const(BigInt(slot.shift)), OP.i64Shl];
  const keep = BigInt.asUintN(64, ~(owned << BigInt(slot.shift)));
  return [
    ...i64Const(owned),
    OP.i64And,
    ...placed,
    ...globalGet(slot.local),
    ...i64Const(keep),
    OP.i64And,
    OP.i64Or,
    ...globalSet(slot.local),
  ];
}

/**
 * Sign-extends the low byte of an i64.
 *
 * An 8-bit comparison is a comparison of the byte's own bits: after `cmp %dh, %dl`, `jl` reads bit 7 of the result and `jb`
 * reads the borrow, so the operand has to arrive sign-extended. Without this a byte holding 0x80 compares as 128 rather than
 * as -128 and every signed branch after a byte comparison goes the wrong way.
 */
function signExtend8(code: number[]): number[] {
  return [
    ...code,
    ...i64Const(0xffn),
    OP.i64And,
    ...i64Const(24n),
    OP.i64Shl,
    ...i64Const(24n),
    OP.i64ShrS,
  ];
}

/** An 8-bit memory operand, which is a byte load this translator does not do yet, refused where it is reached. */
function refuseByteMemory(operand: Operand, instruction: Instruction): void {
  if (operand.kind === "memory" && operand.bits === 8) {
    throw new TranslateError(instruction.offset, "an 8-bit memory operand is not translated yet: the byte load is a width " +
      "the address arithmetic does not yet carry");
  }
}

/**
 * Narrows to 32 bits and widens back unsigned, which is what a 32-bit write does to its result.
 *
 * Unsigned on the way back on purpose: a 32-bit write zeroes the upper half, so `mov $-1, %eax` leaves `rax` holding
 * `0x00000000ffffffff`. The signed form here would be a one-instruction bug visible only on negative values.
 */
const NARROW_32 = OP.i32WrapI64;
const WIDEN_32 = OP.i64ExtendI32U;

function hex(value: bigint): string {
  return `0x${value.toString(16)}`;
}

/**
 * The virtual address a memory operand addresses, or a refusal that says which part of it is the obstacle.
 *
 * Every refusal names what it refused. `[%rcx]` is not "unsupported": it is an address the translator cannot know, because
 * it depends on a register the program will compute later, and saying so is what stops a reader looking for the missing
 * case in the wrong place.
 */
/**
 * The virtual address of the first byte of the body being translated.
 *
 * `functionAddress` is set per body by the program translation and `codeAddress` is where the program starts, so these agree for the
 * entry and differ for everything it calls. A RIP-relative operand is measured from the instruction's own end, which only means
 * something against the body's own address: reading it against the program's would resolve every callee's globals to the entry's
 * neighbourhood.
 */
function bodyBase(options: TranslateOptions): bigint {
  return options.functionAddress ?? options.codeAddress ?? 0n;
}

function virtualAddressOf(operand: Operand, instruction: Instruction, options: TranslateOptions): bigint {
  if (operand.kind !== "memory") {
    throw new TranslateError(instruction.offset, `expected a memory operand, found ${operand.kind}`);
  }
  if (operand.segment !== undefined) {
    throw new TranslateError(
      instruction.offset,
      `a %${operand.segment}: operand needs a segment base, and WASM has no segments: refusing rather than reading the ` +
        `offset as a plain address`,
    );
  }
  if (operand.base !== undefined || operand.index !== undefined) {
    throw new TranslateError(
      instruction.offset,
      "a register-relative address reached the constant path, which is a bug in this translator",
    );
  }
  // RIP-relative is measured from the end of this instruction, which is what the processor adds the displacement to
  if (operand.ripRelative) {
    return bodyBase(options) + BigInt(instruction.offset + instruction.length) + operand.displacement;
  }
  return operand.displacement;
}

/**
 * The WASM that pushes the address of an address-through-a-register operand, or a refusal.
 *
 * A register holds a virtual address at run time, so the subtraction against the image base and the bound check have to
 * happen there too. The bound is not a formality: WASM has one flat memory and no faulting pages, so an address outside the
 * image would read whatever happens to sit at that offset and look entirely legitimate. The check turns that into a trap.
 *
 * The check is unsigned and against `limit - width`, so a negative offset, an offset past the end and a width that would
 * run off the end all fail the same comparison. `i64.gt_u` on a wrapped small negative number is a huge unsigned number,
 * which is exactly what it is.
 */
function runtimeAddressCode(
  operand: Operand,
  instruction: Instruction,
  options: TranslateOptions,
  width: 8 | 16 | 32 | 64,
): number[] {
  if (operand.kind !== "memory") {
    throw new TranslateError(instruction.offset, `expected a memory operand, found ${operand.kind}`);
  }
  const addresses = options.addresses;
  if (addresses === undefined) {
    throw new TranslateError(
      instruction.offset,
      "a memory operand needs the mapped image: pass addresses from image.addressSpaceOf()",
    );
  }
  if (operand.segment !== undefined && operand.segment !== "fs") {
    throw new TranslateError(
      instruction.offset,
      `a %${operand.segment}: operand needs the %${operand.segment} base, which this translator keeps for %fs only: refusing ` +
        "rather than reading the offset as a plain address",
    );
  }
  /*
   * `%fs:` is thread-relative: the sum starts at the thread pointer and every register is an offset inside that block. Without
   * it the base register holds a virtual address, so the image's own base is subtracted from that register before anything is
   * addressed. Either way the result is a virtual address, and the final subtraction is what turns it into an offset.
   */
  const threadRelative = operand.segment === "fs";
  // the address is a sum of parts, and each part is pushed before the add that consumes the two before it
  const parts: number[][] = [];
  if (threadRelative) parts.push([...globalGet(THREAD_GLOBAL)]);
  if (operand.base !== undefined) {
    const register = [...globalGet(localOfRegister(operand.base, instruction))];
    parts.push(threadRelative ? register : [...register, ...i64Const(addresses.baseAddress), OP.i64Sub]);
  }
  if (operand.index !== undefined) {
    const shift = { 1: 0n, 2: 1n, 4: 2n, 8: 3n }[operand.scale];
    const scaled = [...globalGet(localOfRegister(operand.index, instruction))];
    if (shift !== 0n) scaled.push(...i64Const(shift), OP.i64Shl);
    parts.push(scaled);
  }
  const code = [...parts[0]!];
  for (const part of parts.slice(1)) code.push(...part, OP.i64Add);
  if (operand.displacement !== 0n) code.push(...i64Const(operand.displacement), OP.i64Add);
  // under %fs: the sum is a virtual address that has not yet been rebased, so it is turned into an offset here once
  if (threadRelative) code.push(...i64Const(addresses.baseAddress), OP.i64Sub);

  const limit = BigInt((options.memoryPages ?? 1) * PAGE_SIZE) - BigInt(width / 8);
  return [
    ...code,
    OP.localTee, ...unsigned(ADDR),
    ...i64Const(limit), OP.i64GtU,
    OP.ifEmpty, OP.emptyBlockType, OP.unreachable, OP.end,
    OP.localGet, ...unsigned(ADDR),
    OP.i32WrapI64,
  ];
}

/**
 * Resolve every memory operand up front, so the translation never re-derives an address.
 *
 * Doing this in one pass rather than inside each opcode's translation has a second effect worth stating: an operand that
 * cannot be resolved is refused whether or not the instruction around it is one this translator happens to support, so a
 * caller reading errors gets the more fundamental complaint first.
 *
 * An address that is a constant and an address that is a register subtraction both end up as code here, in the same field,
 * so every load and store has one thing to splice and no opinion about which kind it got.
 */
function resolveAddresses(
  instructions: readonly Instruction[],
  options: TranslateOptions,
): readonly Instruction[] {
  return instructions.map((instruction) => {
    if (!instruction.operands.some((operand) => operand.kind === "memory")) return instruction;
    const addresses = options.addresses;
    return {
      ...instruction,
      operands: instruction.operands.map((operand) => {
        if (operand.kind !== "memory") return operand;
        // a register-relative address is only knowable at run time, so it becomes code rather than a number
        if (instruction.op === "lea") return { ...operand, addressCode: leaAddressCode(operand, instruction, options) };
        // %fs: has no base register in the common case, and a thread-relative address is never knowable at translation time
        if (operand.segment === "fs" || operand.base !== undefined || operand.index !== undefined) {
          const width = operand.bits ?? 32;
          return { ...operand, addressCode: runtimeAddressCode(operand, instruction, options, width) };
        }
        const virtual = virtualAddressOf(operand, instruction, options);
        if (addresses === undefined) {
          throw new TranslateError(
            instruction.offset,
            `address ${hex(virtual)} needs the mapped image: pass addresses from image.addressSpaceOf()`,
          );
        }
        const offset = addresses.offsetOf(virtual);
        if (offset === undefined) {
          throw new TranslateError(
            instruction.offset,
            `address ${hex(virtual)} is not mapped by any segment, so there is no byte to access`,
          );
        }
        return { ...operand, wasmOffset: offset, addressCode: i32Const(offset) };
      }),
    };
  });
}

/** The pages the module imports: the image's own, plus whatever was set aside for the stack. */
function totalPages(options: TranslateOptions): number {
  const declared = options.addresses?.pages ?? 0;
  return options.memoryPages ?? (declared + (options.stackPages ?? 0) || 1);
}

/**
 * Where a stack sits when the image occupies the bottom of the imported memory.
 *
 * It grows down from the top page, which keeps it clear of every segment the image maps, and its floor is the image's last
 * page: a push that would descend into the image traps instead of quietly overwriting code. `%rsp` holds a *virtual*
 * address, like every other pointer here, so the top and the floor are both stated in the image's own terms.
 */
function stackBounds(options: TranslateOptions): { top: bigint; floor: bigint } {
  const base = options.addresses?.baseAddress;
  if (base === undefined) {
    throw new TranslateError(0, "a stack needs the mapped image: pass addresses from image.addressSpaceOf()");
  }
  return {
    top: base + BigInt(totalPages(options) * PAGE_SIZE),
    floor: base + BigInt((options.addresses?.pages ?? 0) * PAGE_SIZE),
  };
}

/**
 * The global index of the thread pointer: the one `i64` global the module declares after the register file.
 *
 * `%fs` addressing reads it at run time rather than baking a base into the translation, because `arch_prctl` can point it
 * somewhere else in the middle of a program, and a translator that resolved it once would keep reading the block the process
 * started with. That is the difference between thread-local storage and a guess that happens to work in one thread.
 */
const THREAD_GLOBAL = REGISTERS.length;

/** True when anything in this function uses the stack: a push, a pop, `leave`, or a stack-relative operand. */
function usesStack(instructions: readonly Instruction[]): boolean {
  return instructions.some(
    (instruction) =>
      instruction.op === "push" ||
      instruction.op === "pop" ||
      instruction.op === "leave" ||
      instruction.operands.some(
        (operand) =>
          operand.kind === "memory" &&
          [operand.base, operand.index].some((register) => register === "rsp" || register === "rbp"),
      ),
  );
}

/** The synthetic operand a stack access is expressed as, so it reuses the same address and bound machinery. */
const STACK_OPERAND: Operand = {
  kind: "memory",
  base: "rsp",
  scale: 1,
  displacement: 0n,
  ripRelative: false,
  bits: 64,
};

/**
 * `%rsp` moved by a signed number of bytes, ready for the access that follows it.
 *
 * One `i64.add` in both directions. Writing the negative case as a subtraction would subtract a negative number, which is
 * a push that pushes upwards: the arithmetic is modular, so adding a negative constant is the descent.
 */
function adjustStack(bytes: bigint): number[] {
  const rsp = REGISTERS.indexOf("rsp");
  return [...globalGet(rsp), ...i64Const(bytes), OP.i64Add, ...globalSet(rsp)];
}

/**
 * `push`: the pointer descends first, then the value is written at where it now points.
 *
 * The floor check comes between those two steps, so a push that would leave the stack's own pages traps before it writes
 * rather than after: a write into the image would overwrite the program and look like a wrong answer later.
 */
function translatePush(instruction: Instruction, options: TranslateOptions): number[] {
  const operand = instruction.operands[0];
  if (operand === undefined) {
    throw new TranslateError(instruction.offset, "push needs an operand");
  }
  const value =
    operand.kind === "immediate"
      ? i64Const(BigInt.asIntN(64, operand.value))
      : operand.kind === "reg"
        ? read(operand, instruction)
        : (() => {
            throw new TranslateError(
              instruction.offset,
              "push of a memory operand is not translated yet: it needs the same address machinery as a store",
            );
          })();
  const { floor } = stackBounds(options);
  return [
    ...adjustStack(-8n),
    ...globalGet(REGISTERS.indexOf("rsp")),
    ...i64Const(floor),
    OP.i64LtU,
    OP.ifEmpty,
    OP.emptyBlockType,
    OP.unreachable,
    OP.end,
    ...runtimeAddressCode(STACK_OPERAND, instruction, options, 64),
    ...value,
    OP.i64Store,
    ...memarg(0),
  ];
}

/** `pop`: the value is read where the pointer points, then the pointer rises. */
function translatePop(instruction: Instruction, options: TranslateOptions): number[] {
  const operand = instruction.operands[0];
  if (operand === undefined) {
    throw new TranslateError(instruction.offset, "pop needs an operand");
  }
  if (operand.kind !== "reg") {
    throw new TranslateError(instruction.offset, `pop into ${operand.kind} is not translated yet`);
  }
  return [
    ...runtimeAddressCode(STACK_OPERAND, instruction, options, 64),
    OP.i64Load,
    ...memarg(0),
    ...write(operand, instruction),
    ...adjustStack(8n),
  ];
}

/** `leave`, which is `mov %rbp, %rsp` followed by `pop %rbp`: the frame pointer restores the stack it was given. */
function translateLeave(instruction: Instruction, options: TranslateOptions): number[] {
  const rbp = REGISTERS.indexOf("rbp");
  return [
    ...globalGet(rbp),
    ...globalSet(REGISTERS.indexOf("rsp")),
    // the pop reads through the same address machinery as any other stack access, so it is bound-checked too
    ...runtimeAddressCode(STACK_OPERAND, instruction, options, 64),
    OP.i64Load,
    ...memarg(0),
    ...globalSet(rbp),
    ...adjustStack(8n),
  ];
}

/**
 * The WASM that computes an address for `lea`, which is not the WASM that computes one for a load.
 *
 * A load's operand is rebased (`register - imageBase`) because memory is flat here and 0 there. A register, though, holds a
 * virtual address already, so `lea` into it must leave the sum alone: `lea message(%rip), %rsi` has to put `message`'s virtual
 * address in `%rsi`, and the load that later reads `(%rsi)` is the thing that rebases. Nothing is bound checked either, because
 * an address one past the end is a legal value for `lea` to produce.
 */
function leaAddressCode(operand: Operand, instruction: Instruction, options: TranslateOptions): number[] {
  if (operand.kind !== "memory") {
    throw new TranslateError(instruction.offset, `lea needs a memory operand, found ${operand.kind}`);
  }
  const addresses = options.addresses;
  if (addresses === undefined) {
    throw new TranslateError(instruction.offset, "lea needs the mapped image: pass addresses from image.addressSpaceOf()");
  }
  if (operand.segment !== undefined) {
    throw new TranslateError(
      instruction.offset,
      `lea from a %${operand.segment}: operand is not translated, because the processor itself ignores the override there: ` +
        "`lea %fs:0x10, %rax` loads 0x10, and binutils warns that the segment override is ineffectual",
    );
  }
  if (operand.base === undefined && operand.index === undefined) {
    return [...i64Const(virtualAddressOf(operand, instruction, options))];
  }
  const parts: number[][] = [];
  if (operand.base !== undefined) parts.push([...globalGet(localOfRegister(operand.base, instruction))]);
  if (operand.index !== undefined) {
    const shift = { 1: 0n, 2: 1n, 4: 2n, 8: 3n }[operand.scale];
    const scaled = [...globalGet(localOfRegister(operand.index, instruction))];
    if (shift !== 0n) scaled.push(...i64Const(shift), OP.i64Shl);
    parts.push(scaled);
  }
  const code = [...parts[0]!];
  for (const part of parts.slice(1)) code.push(...part, OP.i64Add);
  if (operand.displacement !== 0n) code.push(...i64Const(operand.displacement), OP.i64Add);
  return code;
}

/** The WASM that pushes a memory operand's address, which `resolveAddresses` has already established. */
function addressOf(operand: Operand, instruction: Instruction): number[] {
  if (operand.kind === "memory" && operand.addressCode !== undefined) return [...operand.addressCode];
  return [...i32Const(wasmOffsetOf(operand, instruction))];
}

/** The resolved offset of a memory operand, which `resolveAddresses` has already established. */
function wasmOffsetOf(operand: Operand, instruction: Instruction): number {
  if (operand.kind !== "memory" || operand.wasmOffset === undefined) {
    throw new TranslateError(
      instruction.offset,
      "a memory operand reached the translation without a resolved address, which is a bug in this translator",
    );
  }
  return operand.wasmOffset;
}

/** A load at the operand's width, as an i64, because every register is an i64 local. */
function loadFromMemory(operand: Operand, width: 16 | 32 | 64, instruction: Instruction): number[] {
  const address = addressOf(operand, instruction);
  if (width === 64) return [...address, OP.i64Load, ...memarg(0)];
  if (width === 16) return [...address, OP.i32Load16U, ...memarg(0), WIDEN_32];
  // a 32-bit load zeroes the upper half of the destination, which is x86's own rule for `movl`
  return [...address, OP.i32Load, ...memarg(0), WIDEN_32];
}

/**
 * A byte of memory, widened to 64 bits with its sign, which is how every byte operand is treated here: a byte's sign lives at bit
 * 7 and every condition a compiler emits after comparing one is about that sign.
 */
function loadByteFromMemory(operand: Operand, instruction: Instruction): number[] {
  return [...addressOf(operand, instruction), OP.i32Load8S, ...memarg(0), OP.i64ExtendI32S];
}

/** A store: the address goes on the stack first, the value on top, which is the order `store` pops them in. */
function storeToMemory(operand: Operand, value: readonly number[], width: 8 | 16 | 32 | 64, instruction: Instruction): number[] {
  const code = { 8: OP.i32Store8, 16: OP.i32Store16, 32: OP.i32Store, 64: OP.i64Store }[width];
  return [...addressOf(operand, instruction), ...value, code, ...memarg(0)];
}

/**
 * The 64 bits of a multiplier, split into halves, as the four partial products a widening multiply is built from.
 *
 * WASM has no 64-by-64 multiply that yields 128 bits, and the product does not fit in an `i64`: so both operands are taken as two
 * 32-bit halves, the four products of those halves are computed (each of which does fit), and the middle column is summed
 * separately because it is the one that carries into both output halves.
 */
function multiplyWide(a: number, b: number): number[] {
  const mask = 0xffffffffn;
  const half = (local: number, shift: bigint): number[] => [
    ...localGet(local),
    ...(shift === 0n ? [] : [...i64Const(shift), OP.i64ShrU]),
    ...i64Const(mask),
    OP.i64And,
  ];
  return [
    // the four partial products: low*low, low*high, high*low, high*high
    ...half(a, 0n), ...half(b, 0n), OP.i64Mul, ...localSet(DIV_REM),
    ...half(a, 0n), ...half(b, 32n), OP.i64Mul, ...localSet(DIV_TAKE),
    ...half(a, 32n), ...half(b, 0n), OP.i64Mul, ...localSet(DIV_CARRY),
    ...half(a, 32n), ...half(b, 32n), OP.i64Mul, ...localSet(DIV_SIGN_Q),
    // the middle column: what the low halves carry into the top half, which is where the product's high 64 bits come from
    ...localGet(DIV_REM), ...i64Const(32n), OP.i64ShrU,
    ...half(DIV_TAKE, 0n), OP.i64Add,
    ...half(DIV_CARRY, 0n), OP.i64Add,
    ...localSet(DIV_HI),
    // the low half is the low dwords, plus the middle column slid up
    ...half(DIV_REM, 0n),
    ...localGet(DIV_HI), ...i64Const(32n), OP.i64Shl, OP.i64Or,
    ...localSet(DIV_Q),
    // and the high half is high*high plus the high dwords of the two mixed products plus the middle column's own carry
    ...localGet(DIV_SIGN_Q),
    ...localGet(DIV_TAKE), ...i64Const(32n), OP.i64ShrU, OP.i64Add,
    ...localGet(DIV_CARRY), ...i64Const(32n), OP.i64ShrU, OP.i64Add,
    ...localGet(DIV_HI), ...i64Const(32n), OP.i64ShrU, OP.i64Add,
    ...localSet(DIV_SIGN_HI),
  ];
}

/** `target = 0 - target` when `condition` holds a non-zero value, which is the wrapping negation an `i64` gives. */
function negateIf(target: number, condition: number[]): number[] {
  return [
    ...condition, OP.i32WrapI64,
    OP.ifEmpty, OP.emptyBlockType,
    ...i64Const(0n), ...localGet(target), OP.i64Sub, ...localSet(target),
    OP.end,
  ];
}

/**
 * Negate the 128-bit value in `hi:lo` when `condition` holds.
 *
 * Two's complement negation of a pair is not two negations: the low half borrows from the high half when it had anything in it, so
 * the high half is `-hi - (lo == 0 ? 1 : 0)` and the low half is `-lo`.
 */
function negateWideIf(hi: number, lo: number, condition: number[]): number[] {
  return [
    ...condition, OP.i32WrapI64,
    OP.ifEmpty,
    OP.emptyBlockType,
    // the high half is its complement plus the carry out of the low half, which is one exactly when the low half was zero
    ...i64Const(0n), ...localGet(hi), OP.i64Sub, ...i64Const(1n), OP.i64Sub,
    ...i64Const(1n), ...i64Const(0n), ...localGet(lo), OP.i64Eqz, OP.select,
    OP.i64Add,
    ...localSet(hi),
    ...i64Const(0n), ...localGet(lo), OP.i64Sub, ...localSet(lo),
    OP.end,
  ];
}

/** A divide error, which the processor raises as a trap rather than as a wrong answer. */
function trapUnless(condition: number[]): number[] {
  return [...condition, OP.ifEmpty, OP.emptyBlockType, OP.unreachable, OP.end];
}

/**
 * The unsigned 128-by-64 division the two divide instructions are built on: a bit at a time, 128 times.
 *
 * The processor answers with a quotient that fits in 64 bits and raises #DE otherwise, which is the check that the dividend's high
 * half is smaller than the divisor. What is left is long division: shift the remainder up, bring in the dividend's next bit, and
 * subtract the divisor when it fits. `take` is computed before the subtraction because it is also the quotient's next bit.
 *
 * The loop is unrolled because 128 is not a variable: a WASM loop would need labels inside whatever blocks the surrounding
 * instruction sits in, and a translation that can nest incorrectly is worse than one that is long.
 */
function divideUnsigned(): number[] {
  const code = trapUnless([...localGet(DIV_HI), ...localGet(DIV_DIVISOR), OP.i64GeU]);
  code.push(...i64Const(0n), ...localSet(DIV_REM), ...i64Const(0n), ...localSet(DIV_Q));
  for (let bit = 127; bit >= 0; bit -= 1) {
    const from = bit >= 64 ? DIV_HI : DIV_LO;
    code.push(
      // the bit the shift drops is the one that makes the remainder exceed 64 bits, and the subtraction still works when it is set
      ...localGet(DIV_REM), ...i64Const(63n), OP.i64ShrU, ...localSet(DIV_CARRY),
      ...localGet(DIV_REM), ...i64Const(1n), OP.i64Shl,
      ...localGet(from), ...i64Const(BigInt(bit % 64)), OP.i64ShrU, ...i64Const(1n), OP.i64And,
      OP.i64Or, ...localSet(DIV_REM),
      // `take` is a 0 or a 1: the comparison is widened, and the carry is already one because a shift of a 64-bit value by 63 is
      ...localGet(DIV_REM), ...localGet(DIV_DIVISOR), OP.i64GeU, OP.i64ExtendI32U, ...localGet(DIV_CARRY), OP.i64Or,
      ...localSet(DIV_TAKE),
      // subtracting `divisor * take` is the trial subtraction: the divisor, or nothing at all
      ...localGet(DIV_REM), ...localGet(DIV_DIVISOR), ...localGet(DIV_TAKE), OP.i64Mul, OP.i64Sub, ...localSet(DIV_REM),
      ...localGet(DIV_Q), ...i64Const(1n), OP.i64Shl, ...localGet(DIV_TAKE), OP.i64Or, ...localSet(DIV_Q),
    );
  }
  return code;
}

/** True for a register name that is one half of a vector register, as opposed to a general or byte register. */
function isVectorHalf(operand: Operand): boolean {
  return operand.kind === "reg" && (operand.name.endsWith(".lo") || operand.name.endsWith(".hi"));
}

/** The other half of a vector register, which is where a write of the whole register puts the bits it did not use. */
function otherHalf(operand: Operand, instruction: Instruction): Operand {
  if (operand.kind !== "reg") throw new TranslateError(instruction.offset, "a vector register is a register operand");
  const { name, bits } = operand;
  const swapped = name.endsWith(".lo") ? `${name.slice(0, -3)}.hi` : name.endsWith(".hi") ? `${name.slice(0, -3)}.lo` : undefined;
  if (swapped === undefined) throw new TranslateError(instruction.offset, `${name} is not half of a vector register`);
  return { kind: "reg", name: swapped as RegisterName, bits };
}

function translateMov(instruction: Instruction): number[] {
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, "mov needs two operands");
  }
  if (destination.kind === "memory") {
    return translateMovToMemory(destination, source, instruction);
  }
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `a mov into a ${destination.kind} is not translated`);
  }
  const width = destination.bits;
  refuseByteMemory(source, instruction);
  if (source.kind === "immediate") {
    // a narrow destination keeps only its own bits, so this masks rather than sign-extends: `mov $-1, %eax` leaves
    // 0xffffffff, and `mov $-1, %dh` leaves 0xff in the second byte of `%rdx`
    const value = width === 64 ? source.value : BigInt.asUintN(width, source.value);
    return [...i64Const(value), ...write(destination, instruction)];
  }
  if (source.kind === "memory") {
    const bits = source.bits;
    if (bits !== 16 && bits !== 32 && bits !== 64) {
      throw new TranslateError(instruction.offset, `a ${bits}-bit memory load is not translated yet`);
    }
    // the load already yields the destination's width, so no narrowing of a register is involved
    return [...loadFromMemory(source, bits, instruction), ...write(destination, instruction)];
  }
  const code = read(source, instruction);
  if (width === 32) return [...code, NARROW_32, WIDEN_32, ...write(destination, instruction)];
  // a 16-bit source is already masked to its own half by `read`, and `write` merges it into the destination's
  return [...code, ...write(destination, instruction)];
}

/**
 * A `mov` into memory.
 *
 * The width comes from the source rather than the destination, because the destination does not carry one: `movb $1,
 * 0x404000` and `movq $1, 0x404000` differ only in the opcode and register, and the decoder has already resolved that into
 * the source operand's width. A store whose width the decoder could not pin down is refused rather than guessed at, since
 * writing four bytes where one was meant corrupts two neighbours silently.
 */
function translateMovToMemory(destination: Operand, source: Operand, instruction: Instruction): number[] {
  if (source.kind === "memory") {
    throw new TranslateError(instruction.offset, "a mov between two memory operands is not an x86 instruction");
  }
  const width = source.kind === "reg" ? source.bits : memoryStoreWidth(instruction);
  if (width !== 8 && width !== 16 && width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit store is not translated yet`);
  }
  if (source.kind === "immediate") {
    const value = width === 64 ? source.value : BigInt.asUintN(width, source.value);
    const code = width === 64 ? i64Const(value) : i32Const(Number(value));
    return storeToMemory(destination, code, width, instruction);
  }
  const value = source.kind === "reg" ? read(source, instruction) : [];
  if (width === 32 || width === 8 || width === 16) {
    // a store reads the low bits of the register, so narrowing is a wrap rather than a reinterpretation
    return storeToMemory(destination, [...value, NARROW_32], width, instruction);
  }
  return storeToMemory(destination, value, width, instruction);
}

/**
 * The width of a store whose source is an immediate.
 *
 * x86 encodes `movb`/`movw`/`movl`/`movq` into memory as separate opcodes, so the width belongs to the instruction rather
 * than to any operand: the decoder keeps it on the destination for exactly this reason. Without it the width is unknown,
 * and guessing four bytes where one was meant would corrupt three neighbours while validating perfectly.
 */
function memoryStoreWidth(instruction: Instruction): 8 | 16 | 32 | 64 {
  const destination = instruction.operands[0];
  const bits = destination?.kind === "memory" ? destination.bits : undefined;
  if (bits === undefined) {
    throw new TranslateError(
      instruction.offset,
      "a store of an immediate whose width the encoding did not determine is not translated",
    );
  }
  return bits;
}

/** The arithmetic, at both widths: a 32-bit operation happens in i32 and widens back, so it wraps at 32 bits. */
const BINARY = {
  add: { bits64: OP.i64Add, bits32: OP.i32Add },
  sub: { bits64: OP.i64Sub, bits32: OP.i32Sub },
  xor: { bits64: OP.i64Xor, bits32: OP.i32Xor },
  and: { bits64: OP.i64And, bits32: OP.i32And },
  or: { bits64: OP.i64Or, bits32: OP.i32Or },
  imul: { bits64: OP.i64Mul, bits32: OP.i32Mul },
} as const;

type BinaryOp = keyof typeof BINARY;

/**
 * The three shifts, which differ only in which direction the bits leave and whether the sign comes with them.
 *
 * `shl` moves them up, `shr` moves them down filling with zero, and `sar` moves them down filling with the sign bit, which is the one
 * a compiler picks for a signed division by a power of two. All three take their count from an immediate or from `%cl`.
 */
const SHIFTS_TRANSLATED = {
  shl: { bits64: OP.i64Shl, bits32: OP.i32Shl },
  shr: { bits64: OP.i64ShrU, bits32: OP.i32ShrU },
  sar: { bits64: OP.i64ShrS, bits32: OP.i32ShrS },
} as const;

type ShiftOp = keyof typeof SHIFTS_TRANSLATED;

/**
 * A shift of a register by an immediate or by `%cl`.
 *
 * The count is masked to five bits at 32 and six at 64, because that is what the processor does with it: `sar %cl` with 33 in `%cl`
 * shifts by one. `wasm` masks a count the same way, so the masking here is stated rather than relied on, since the flags below are
 * about what the processor's own definition says.
 *
 * The result sets the flags a comparison of it against zero would: `sar` of a word to zero is a common way to test the sign, and the
 * zero and sign conditions read the value that was written. The carry is the last bit shifted out, which is a fact about the bits
 * that left rather than about the result, so it is not modelled here and the conditions that read it refuse by name.
 */
/**
 * `bsf` and `bsr`: the index of the lowest or highest set bit of the source, written into the destination.
 *
 * Neither has a WASM instruction, and both are built from one: `ctz` counts the bits after the lowest set one, which is its index,
 * and the highest index is `63 - clz`. The case that neither count gets right is a zero source: `ctz` answers 64 and `clz` answers
 * 64, where the processor leaves the destination exactly as it was and sets the zero flag. So the destination is picked between its
 * own value and the index by a `select`, and the flag locals hold the source against zero rather than the result.
 */
function translateBitScan(instruction: Instruction): number[] {
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, `a ${instruction.op} needs two operands`);
  }
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} into a ${destination.kind} is not an encoding: the destination is a register`);
  }
  const width = destination.bits;
  if (width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} is not translated: only 32 and 64`);
  }
  const sourceBits = source.kind === "reg" || source.kind === "memory" ? source.bits : undefined;
  if (sourceBits !== width) {
    throw new TranslateError(
      instruction.offset,
      `a ${instruction.op} from a ${sourceBits ?? source.kind} operand into a ${width}-bit destination is not an encoding`,
    );
  }
  const raw = source.kind === "memory" ? loadFromMemory(source, width, instruction) : read(source, instruction);
  // a 32-bit operand is read from a 64-bit local, so the bits above its width are not part of what the scan walks
  const value = width === 32 ? [...raw, ...i64Const(0xffffffffn), OP.i64And] : raw;
  const index = instruction.op === "bsf"
    ? [...value, OP.i64Ctz]
    : [...i64Const(63n), ...value, OP.i64Clz, OP.i64Sub];
  return [
    ...value,
    ...localSet(CMP_A),
    ...i64Const(0n),
    ...localSet(CMP_B),
    // the destination is its own value when the source is zero, and the index otherwise: `select` picks the first when the
    // condition is not zero, so the source being zero is the condition it gets
    ...read(destination, instruction),
    ...index,
    ...value,
    OP.i64Eqz,
    OP.select,
    ...write(destination, instruction),
  ];
}

function translateShift(instruction: Instruction): number[] {
  const op = instruction.op as ShiftOp;
  const [destination, count] = instruction.operands;
  if (destination === undefined || count === undefined) {
    throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
  }
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} into ${destination.kind} is not translated yet`);
  }
  const width = destination.bits;
  if (width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} is not translated yet`);
  }
  if (count.kind === "memory") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} whose count is in memory is not translated yet`);
  }
  const opcodes = SHIFTS_TRANSLATED[op];
  const masked = [...operandCode64(count, instruction), ...i64Const(BigInt(width - 1)), OP.i64And];
  const shifted =
    width === 64
      ? [...read(destination, instruction), ...masked, opcodes.bits64]
      : [
          ...read(destination, instruction),
          NARROW_32,
          ...masked,
          NARROW_32,
          opcodes.bits32,
          WIDEN_32,
        ];
  return [
    ...shifted,
    OP.localTee,
    ...unsigned(CMP_A),
    ...i64Const(0n),
    OP.localSet,
    ...unsigned(CMP_B),
    ...write(destination, instruction),
  ];
}

/**
 * An operand at the width the operation happens at, ready to sit next to the destination on the stack.
 *
 * A 32-bit register source must be narrowed as well as the destination. Narrowing only the destination is the bug this
 * comment exists for: the destination arrives as i32 and the source as i64, and the module fails to validate, which is
 * loud. The quiet version of the same mistake is a sign-extended source, which computes the wrong answer and validates.
 */
function operandCode(operand: Operand, width: 32 | 64, instruction: Instruction): number[] {
  if (operand.kind === "immediate") {
    return width === 32 ? i32Const(Number(BigInt.asIntN(32, operand.value))) : i64Const(operand.value);
  }
  if (operand.kind === "memory") {
    // the load is already the operation's width, so there is nothing to narrow
    const address = addressOf(operand, instruction);
    if (width === 64) return [...address, OP.i64Load, ...memarg(0)];
    return [...address, OP.i32Load, ...memarg(0)];
  }
  const code = read(operand, instruction);
  return width === 32 ? [...code, NARROW_32] : code;
}

/**
 * An operand as an i64, which is what `cmp` and `test` store.
 *
 * The flag locals hold whole 64-bit values even for a 32-bit comparison, and the narrowing happens when the condition is
 * evaluated. That keeps the four ways an operand can arrive (register, immediate, and their widths) out of the branch
 * path: the low 32 bits of an i64 are the 32 bits a 32-bit compare looks at, whichever form they came in.
 */
function operandCode64(operand: Operand, instruction: Instruction): number[] {
  if (operand.kind === "immediate") return i64Const(operand.value);
  if (operand.kind === "memory") {
    // the comparison's own width decides, and a memory operand carries it: a byte of memory is not a 64-bit value
    if (operand.bits === 8) return loadByteFromMemory(operand, instruction);
    const width = operand.bits === 16 ? 16 : operand.bits === 32 ? 32 : 64;
    return loadFromMemory(operand, width, instruction);
  }
  return read(operand, instruction);
}

/**
 * The arithmetic instructions whose result also sets the flags a branch reads.
 *
 * `cmp` and `test` were the only ways a condition could be set up, which is enough for a hand-written program and not enough
 * for a compiler's: gcc compares a stack canary with `sub %fs:0x28, %rdx` followed by `jne`, and refuses nothing. `imul` is
 * left out because its flags are not the same ones: it reports overflow in a carry flag the others set by addition.
 */
const FLAG_SETTING: ReadonlySet<string> = new Set(["add", "sub", "and", "or", "xor"]);

/**
 * `not` and `neg`, whose operand is both source and destination.
 *
 * `not` leaves the flags alone, which is not the same as clearing them: a branch after it reads what the previous
 * instruction set. `neg` reports its result, so only the zero and sign flags are modelled, and its borrow is not.
 */
function translateUnary(instruction: Instruction): { code: number[]; width?: 16 | 32 | 64 } {
  const [destination] = instruction.operands;
  if (destination === undefined) {
    throw new TranslateError(instruction.offset, `${instruction.op} needs an operand`);
  }
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} in memory is not translated yet`);
  }
  const width = destination.bits;
  if (width !== 8 && width !== 16 && width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} is not translated yet`);
  }
  const ones = width === 8 ? 0xffn : width === 16 ? 0xffffn : width === 32 ? 0xffffffffn : 0xffffffffffffffffn;
  // a narrow result is reduced to its own half here, so the value that goes into the local and the value that is reported to the
  // flags are the same one, and a sixteen-bit one is kept masked rather than sign-extended for the reason a comparison is
  const narrowed = (code: number[]): number[] =>
    width === 32 ? [...code, NARROW_32, WIDEN_32] : width === 16 ? [...code, ...i64Const(0xffffn), OP.i64And] : code;
  if (instruction.op === "not") {
    return { code: [...narrowed([...read(destination, instruction), ...i64Const(ones), OP.i64Xor]), ...write(destination, instruction)] };
  }
  // neg is 0 - value, and its flags describe that difference: `neg %eax` with a zero operand sets ZF
  const byte = width === 8
    ? signExtend8([...i64Const(0n), ...read(destination, instruction), OP.i64Sub, ...i64Const(0xffn), OP.i64And])
    : narrowed([...i64Const(0n), ...read(destination, instruction), OP.i64Sub]);
  return {
    code: [
      ...byte,
      OP.localTee,
      ...unsigned(CMP_A),
      ...i64Const(0n),
      OP.localSet,
      ...unsigned(CMP_B),
      ...write(destination, instruction),
    ],
    width: width === 8 ? 64 : width,
  };
}

/**
 * A binary operation whose destination is memory: read the word, fold the register into it, write it back.
 *
 * The address is pushed before the value every time, because that is the order `store` pops them in, and the flags are set the way
 * the register form sets them: `sub` keeps its two operands, because its flags describe the difference the way a `cmp` does, and
 * every other operation keeps its result against zero, because `or`'s zero flag describes the value it just wrote.
 *
 * A byte into memory refuses by name: a byte operand is sign-extended here for the sake of the flags, and a store that has to write
 * one byte of a word it read the same way is a separate question from this one.
 */
function translateIntoMemory(
  instruction: Instruction,
  op: BinaryOp,
  destination: Operand,
  source: Operand,
): number[] {
  if (destination.kind !== "memory") {
    throw new TranslateError(instruction.offset, `${instruction.op} into a ${destination.kind} is not a memory operation`);
  }
  const width = destination.bits;
  if (width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} into memory is not translated yet`);
  }
  const opcodes = BINARY[op];
  const load = loadFromMemory(destination, width, instruction);
  const address = (): number[] => addressOf(destination, instruction);
  const store = width === 64 ? OP.i64Store : OP.i32Store;
  // the value the store writes: a 32-bit store pops an i32, and everything here computes at 64
  const stored = (code: number[]): number[] => (width === 64 ? code : [...code, NARROW_32]);
  const combine =
    width === 64
      ? [...load, ...operandCode64(source, instruction), opcodes.bits64]
      : [...load, NARROW_32, ...operandCode64(source, instruction), NARROW_32, opcodes.bits32, WIDEN_32];
  if (!FLAG_SETTING.has(op)) return [...address(), ...stored(combine), store, ...memarg(0)];
  if (op !== "sub") {
    /*
     * The result is kept in `CMP_A` and compared against zero the way the register form does, then written: what the stack holds at
     * the `store` is the address and the value, in that order, and the comparison locals are already filled by then.
     */
    return [
      ...combine,
      ...localSet(CMP_A),
      ...i64Const(0n),
      ...localSet(CMP_B),
      ...address(),
      ...stored([...localGet(CMP_A)]),
      store,
      ...memarg(0),
    ];
  }
  /*
   * `sub %eax,(%rbx)` subtracts the register from the word, so its flags describe the word minus the register: the two are kept in the
   * comparison locals first, the difference is rebuilt from them, and what is left behind is the pair a `cmp` would have kept.
   */
  /*
   * The register is kept at its own 64-bit width, exactly as the register form keeps it: a comparison narrows both operands itself
   * when it reads them back, so narrowing here would leave a 32-bit value in a 64-bit local and the module would not validate.
   */
  const operand = operandCode64(source, instruction);
  return [
    ...load,
    ...localSet(CMP_A),
    ...operand,
    ...localSet(CMP_B),
    ...address(),
    ...stored([...localGet(CMP_A), ...localGet(CMP_B), OP.i64Sub]),
    store,
    ...memarg(0),
  ];
}

function translateBinary(instruction: Instruction): number[] {
  const op = instruction.op as BinaryOp;
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
  }
  /*
   * The other direction of the same instruction: `or %eax,(%rbx)` folds a register into a word in memory. Until a program needed it,
   * every memory operand here was a source, so the read-modify-write path did not exist and `0x40a360`'s callee stopped on it.
   */
  if (destination.kind === "memory") return translateIntoMemory(instruction, op, destination, source);
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} into memory is not translated yet`);
  }
  const width = destination.bits;
  if (width !== 8 && width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} is not translated yet`);
  }
  refuseByteMemory(source, instruction);
  const opcodes = BINARY[op];
  if (width === 8) {
    /*
     * A byte operation is done at 64 bits and masked back to eight, which wraps exactly where the processor wraps. The flags
     * are shaped the same way a byte comparison's are, sign-extended, because a byte result of 0x80 is -128 to every branch
     * that reads it.
     */
    const value = [...read(destination, instruction), ...operandCode64(source, instruction), opcodes.bits64];
    const byte = [...value, ...i64Const(0xffn), OP.i64And];
    if (!FLAG_SETTING.has(op)) {
      return [...byte, ...write(destination, instruction)];
    }
    if (op === "sub") {
      const operand = operandCode64(source, instruction);
      return [
        ...signExtend8(read(destination, instruction)),
        ...localSet(CMP_A),
        ...signExtend8(operand),
        ...localSet(CMP_B),
        ...i64Const(0n),
        ...localGet(CMP_A),
        ...localGet(CMP_B),
        OP.i64Sub,
        ...i64Const(0xffn),
        OP.i64And,
        ...write(destination, instruction),
      ];
    }
    return [
      ...signExtend8(byte),
      OP.localTee,
      ...unsigned(CMP_A),
      ...i64Const(0n),
      ...localSet(CMP_B),
      ...write(destination, instruction),
    ];
  }
  if (FLAG_SETTING.has(op)) {
    /*
     * Both operands are kept in the comparison locals before the result overwrites the destination, so a branch after the
     * arithmetic reads the same two numbers `cmp` would have kept. Neither is left on the stack for the result to find again:
     * the sum is rebuilt from the two locals, so the operand stack holds nothing but the result when `write` takes it.
     */
    /*
     * The two halves of this are different questions, and only one of them is a comparison.
     *
     * `sub` sets its flags from `destination - source`, which is exactly what `cmp source, destination` computes, so the two
     * operands are kept and every condition means what it means after a `cmp`. The others set their flags from the *result*:
     * after `add $6, %rax` the zero and sign flags describe twelve, not the six and six that went in, so what is kept is the
     * result and a zero to compare it against. Keeping the operands there would answer a question nobody asked, and the wrong
     * answer is a branch that disagrees with the processor.
     */
    const keeps = (code: number[], local: number): number[] => [...code, OP.localSet, ...unsigned(local)];
    if (op !== "sub") {
      const result =
        width === 64
          ? [
              ...read(destination, instruction),
              ...operandCode64(source, instruction),
              opcodes.bits64,
            ]
          : [
              ...read(destination, instruction),
              NARROW_32,
              ...operandCode64(source, instruction),
              NARROW_32,
              opcodes.bits32,
              WIDEN_32,
            ];
      return [
        ...result,
        OP.localTee,
        ...unsigned(CMP_A),
        ...i64Const(0n),
        OP.localSet,
        ...unsigned(CMP_B),
        ...write(destination, instruction),
      ];
    }
    const left = keeps(read(destination, instruction), CMP_A);
    const right = keeps(operandCode64(source, instruction), CMP_B);
    if (width === 64) {
      return [...left, ...right, ...localGet(CMP_A), ...localGet(CMP_B), opcodes.bits64, ...write(destination, instruction)];
    }
    // both are kept at 64 bits, so a 32-bit operation narrows them on the way back in, exactly as `cmp` leaves them
    return [
      ...left,
      ...right,
      ...localGet(CMP_A),
      NARROW_32,
      ...localGet(CMP_B),
      NARROW_32,
      opcodes.bits32,
      WIDEN_32,
      ...write(destination, instruction),
    ];
  }
  const operand = operandCode(source, width, instruction);
  if (width === 64) {
    return [...read(destination, instruction), ...operand, opcodes.bits64, ...write(destination, instruction)];
  }
  return [
    ...read(destination, instruction),
    NARROW_32,
    ...operand,
    opcodes.bits32,
    WIDEN_32,
    ...write(destination, instruction),
  ];
}

/** `cmp` keeps both operands; `test` keeps the mask, so every condition is one comparison against the other local. */
/**
 * A conditional move: the destination keeps its value unless the condition holds, in which case the source is written into it.
 *
 * The flags are not touched, which is the point of the instruction and the reason it cannot be translated as a branch around a
 * `mov`: a `cmov` sits between a comparison and the branch that uses the same flags.
 */
function translateConditionalMove(instruction: Instruction): number[] {
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, "a conditional move needs a destination and a source");
  }
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, "a conditional move writes a register, and x86 has no memory destination for it");
  }
  const width: 32 | 64 = destination.bits === 32 ? 32 : 64;
  const value =
    source.kind === "memory"
      ? loadFromMemory(source, width, instruction)
      : width === 32
        ? [...read(source, instruction), NARROW_32]
        : read(source, instruction);
  // a 32-bit write clears the upper half of the register, so the value arrives in the local zero-extended
  const widened = width === 32 ? [...value, OP.i64ExtendI32U] : value;
  return [...widened, ...write(destination, instruction)];
}

/**
 * The accumulator each width of `cmpxchg` reads and, on a mismatch, writes: the register it never touches on a match.
 *
 * A thirty-two-bit operand names the sixty-four-bit register and carries its own width, which is the convention every other operand
 * here uses; only the word has a name of its own.
 */
const ACCUMULATOR: Readonly<Record<16 | 32 | 64, RegisterName>> = { 16: "ax", 32: "rax", 64: "rax" };

/**
 * `cmpxchg`: one instruction that is a comparison and a choice.
 *
 * It compares the accumulator with the destination, and then exactly one of two writes happens. Equal, the source goes where the
 * destination was and the accumulator keeps its value; not equal, the destination's old value goes into the accumulator and the
 * destination keeps its own. Both writes are on the same path and neither is speculative, which is what makes the atomicity of the
 * prefix unobservable here: with one thread there is nothing between the comparison and the write to interleave with.
 *
 * The flags are the comparison's own, which is why the two values are left in the comparison locals: `jne` after it reads them, and
 * so does the mismatch path, whose value is the second of the two.
 */
function translateCompareExchange(instruction: Instruction): { code: number[]; width: 16 | 32 | 64 } {
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, "cmpxchg needs two operands");
  }
  if (destination.kind !== "memory" && destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `cmpxchg into a ${destination.kind} is not an encoding`);
  }
  const width = destination.bits;
  if (width !== 16 && width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, "an 8-bit cmpxchg is not translated yet: it compares the low byte of the accumulator");
  }
  const accumulator: Operand = { kind: "reg", name: ACCUMULATOR[width], bits: width };
  // a sixteen-bit value is kept masked so the unsigned conditions are right, and a thirty-two-bit one is narrowed through i32 so
  // that a write of it zero-extends, the same rule every other operand here follows
  const half = (code: number[]): number[] =>
    width === 16 ? [...code, ...i64Const(0xffffn), OP.i64And] : width === 32 ? [...code, NARROW_32, WIDEN_32] : code;
  const current =
    destination.kind === "memory" ? loadFromMemory(destination, width, instruction) : read(destination, instruction);
  const replacement = half(read(source, instruction));
  return {
    code: [
      ...half(read(accumulator, instruction)),
      ...localSet(CMP_A),
      ...half(current),
      ...localSet(CMP_B),
      ...localGet(CMP_A),
      ...localGet(CMP_B),
      OP.i64Eq,
      OP.ifEmpty,
      OP.emptyBlockType,
      ...(destination.kind === "memory"
        ? storeToMemory(destination, width === 64 ? replacement : [...replacement, NARROW_32], width, instruction)
        : [...replacement, ...write(destination, instruction)]),
      OP.elseOp,
      ...localGet(CMP_B),
      ...write(accumulator, instruction),
      OP.end,
    ],
    width,
  };
}

function translateFlags(instruction: Instruction): { code: number[]; width: 16 | 32 | 64 } {
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
  }
  /*
   * A comparison against memory is how a compiled program tests a feature flag: `testb $0x2, 0x9...(%rip)`. Nothing is written,
   * so the operand only has to be read at its own width, and a byte of it is read as signed like every other byte here.
   */
  if (destination.kind !== "reg" && destination.kind !== "memory") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} against a ${destination.kind} is not translated`);
  }
  const width = destination.bits;
  if (width !== 8 && width !== 16 && width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} is not translated yet`);
  }
  /*
   * A byte comparison is evaluated at 64 bits on sign-extended bytes, so `width` is reported as the width the comparison is
   * *done* at rather than the width the instruction names. A word is different: its operands are masked to sixteen bits and kept
   * that way, because a sign-extended word gets the unsigned conditions wrong. That is the same convention a 32-bit comparison already uses: its
   * operands are kept as i64 and narrowed when the condition is evaluated, and here the narrowing happens on the way in
   * because a byte's sign lives at bit 7 and not at bit 31.
   */
  const shape = (code: number[]): number[] =>
    width === 8 ? signExtend8(code) : width === 16 ? [...code, ...i64Const(0xffffn), OP.i64And] : code;
  const left =
    destination.kind === "memory"
      ? destination.bits === 8
        ? loadByteFromMemory(destination, instruction)
        : loadFromMemory(destination, destination.bits === 16 ? 16 : destination.bits === 32 ? 32 : 64, instruction)
      : read(destination, instruction);
  if (instruction.op === "test") {
    // `test a, b` is `and` with the result discarded: the flags come from a & b, and comparing the mask against zero
    // reproduces every condition a compiler emits after it, jl and js included
    const mask = shape([...left, ...operandCode64(source, instruction), OP.i64And]);
    return {
      code: [...mask, ...localSet(CMP_A), ...i64Const(0n), ...localSet(CMP_B)],
      width: width === 8 ? 64 : width,
    };
  }
  return {
    code: [...shape(left), ...localSet(CMP_A), ...shape(operandCode64(source, instruction)), ...localSet(CMP_B)],
    width: width === 8 ? 64 : width,
  };
}

/** Condition suffixes the encoding numbers, mapped to the comparison each one is. */
const COMPARISONS = {
  e: "eq",
  ne: "ne",
  l: "ltS",
  le: "leS",
  g: "gtS",
  ge: "geS",
  b: "ltU",
  be: "leU",
  a: "gtU",
  ae: "geU",
} as const;

type Comparison = (typeof COMPARISONS)[keyof typeof COMPARISONS];

const COMPARE_OPCODES: Record<32 | 64, Record<Comparison, number>> = {
  32: {
    eq: OP.i32Eq, ne: OP.i32Ne, ltS: OP.i32LtS, ltU: OP.i32LtU, gtS: OP.i32GtS,
    gtU: OP.i32GtU, leS: OP.i32LeS, leU: OP.i32LeU, geS: OP.i32GeS, geU: OP.i32GeU,
  },
  64: {
    eq: OP.i64Eq, ne: OP.i64Ne, ltS: OP.i64LtS, ltU: OP.i64LtU, gtS: OP.i64GtS,
    gtU: OP.i64GtU, leS: OP.i64LeS, leU: OP.i64LeU, geS: OP.i64GeS, geU: OP.i64GeU,
  },
};

/** A sixteen-bit value sign-extended from its own half, so a signed comparison is about bit 15 rather than bit 63. */
function signExtend16(value: number[]): number[] {
  return [...value, ...i64Const(48n), OP.i64Shl, ...i64Const(48n), OP.i64ShrS];
}

/**
 * Both operands of the last comparison, at the width it was made, ready for one comparison opcode.
 *
 * A sixteen-bit comparison is kept masked to its own half, which is what makes the unsigned conditions right: `0x8000` is above
 * `0x7fff` and both are below every value a wider register holds. A signed condition needs the sign of bit 15 and not of bit 63, so
 * the two bits are swapped first and the 64-bit signed opcodes then answer it.
 */
function comparisonOperands(width: 16 | 32 | 64, signed = false): number[] {
  if (width === 64) return [...localGet(CMP_A), ...localGet(CMP_B)];
  if (width === 16) {
    const left = [...localGet(CMP_A)];
    const right = [...localGet(CMP_B)];
    return signed ? [...signExtend16(left), ...signExtend16(right)] : [...left, ...right];
  }
  return [...localGet(CMP_A), NARROW_32, ...localGet(CMP_B), NARROW_32];
}

/**
 * A condition as an i32 on the stack.
 *
 * `js` and `jns` are the only two that need the subtraction rather than a comparison, because the sign flag is the sign
 * of the difference: `a - b` is pushed and compared against zero, which is the definition rather than a shortcut.
 */
function conditionCode(suffix: string, width: 16 | 32 | 64): number[] {
  if (suffix === "s" || suffix === "ns") {
    const difference =
      width === 64
        ? [...comparisonOperands(64), OP.i64Sub, ...i64Const(0n), OP.i64LtS]
        : width === 16
          ? [...comparisonOperands(16, true), OP.i64Sub, ...i64Const(0n), OP.i64LtS]
          : [...comparisonOperands(32), OP.i32Sub, ...i32Const(0), OP.i32LtS];
    return suffix === "s" ? difference : [...difference, OP.i32Eqz];
  }
  const comparison = COMPARISONS[suffix as keyof typeof COMPARISONS];
  if (comparison === undefined) {
    throw new Error(`unreachable: ${suffix} has no comparison`);
  }
  // `ltS`, `leS`, `gtS` and `geS` are the four that are about the sign; eq, ne and the unsigned four are about the bits
  const signed = comparison.endsWith("S");
  return [...comparisonOperands(width, signed), COMPARE_OPCODES[width === 32 ? 32 : 64][comparison]];
}

/**
 * The comparison a condition reads, from the flags as they stand.
 *
 * `bt` is the reason this exists apart from `conditionCode`: it writes the carry and leaves every other flag undefined, so a
 * condition that reads one of those has no answer here and is refused rather than answered from whatever the last comparison left
 * in the locals. `b` and `ae` are the unsigned-below pair, which is how the carry is expressed: the same pair a `cmp` produces.
 */
function flagCondition(suffix: string, flags: Flags, instruction: Instruction): number[] {
  if (flags.kind === "bit" && suffix !== "b" && suffix !== "ae") {
    throw new TranslateError(
      instruction.offset,
      `a ${instruction.op} after a bit test reads a flag the bit test does not define: bt writes the carry and leaves the rest ` +
        "undefined, so only the carry conditions, jb and jae, can be read after it",
    );
  }
  if (flags.kind === "zero" && !ZERO_CONDITIONS.has(suffix)) {
    throw new TranslateError(
      instruction.offset,
      `a ${instruction.op} after a bit scan reads a flag the scan does not define: bsf and bsr set the zero flag when no bit is set ` +
        "and leave the sign, the parity, the carry and the overflow undefined, so only je and jne can be read after one",
    );
  }
  if (flags.kind === "shift" && !SHIFT_CONDITIONS.has(suffix)) {
    throw new TranslateError(
      instruction.offset,
      `a ${instruction.op} after a shift reads a flag the shift does not define here: the zero and sign conditions describe the ` +
        "shifted value, but the carry is the last bit shifted out, which is a fact about the bits that left and is not modelled",
    );
  }
  return conditionCode(suffix, flags.width);
}

/** The conditions a shift answers: what the result it wrote is, and not what left it. */
const SHIFT_CONDITIONS: ReadonlySet<string> = new Set(["e", "ne", "z", "nz", "s", "ns"]);

/** The conditions a bit scan answers: whether it found a bit, and nothing about the value it found. */
const ZERO_CONDITIONS: ReadonlySet<string> = new Set(["e", "ne", "z", "nz"]);

/** A region's exit points: the instruction index of its end, and how many blocks a `br` must cross to reach it. */
type Scope = Map<number, number>;

interface Flags {
  set: boolean;
  width: 16 | 32 | 64;
  /** Whether the comparison was a subtraction, whose borrow is the carry `sbb` reads; a `test` has no carry to give. */
  borrows?: boolean;
  /**
   * What defined them. A comparison defines every condition a compiler reads; a bit test defines the carry and leaves the rest
   * undefined, so the conditions that read them are refused instead of being answered from a value that is not there.
   */
  kind?: "bit" | "shift" | "zero" | undefined;
}

function indexOfOffset(instructions: readonly Instruction[], offset: number): number {
  const index = instructions.findIndex((instruction) => instruction.offset === offset);
  if (index < 0) {
    throw new TranslateError(offset, "a jump targets an offset that is not an instruction boundary in this block");
  }
  return index;
}

/** The absolute target of a branch, from its displacement and its own end. */
function branchTarget(instruction: Instruction): number {
  const operand = instruction.operands[0];
  if (operand === undefined || operand.kind !== "immediate") {
    throw new TranslateError(instruction.offset, `${instruction.op} has no displacement to follow`);
  }
  return instruction.offset + instruction.length + Number(operand.value);
}

/**
 * The join of the if/else a compiler emits: `jcc else`, the then-arm, `jmp join`, the else-arm, `join:`.
 *
 * It is worth recognising as a shape rather than translating branch by branch, because the two forms differ in what a
 * branch can reach. A conditional jump is an `if` around the code it skips, and the code under it falls through. But a
 * `jmp` out of that code has to leave the whole construct, which plain nesting cannot express unless the construct is
 * an `if`/`else` with the join as its end. Translating the pattern as one if/else is both simpler and correct, where
 * the naive nesting silently ran the else arm after a then arm that had already jumped past it.
 */
function findJoin(
  instructions: readonly Instruction[],
  from: number,
  limit: number,
  target: number,
  regionEnd: number,
): { jmp: number; join: number } | undefined {
  for (let index = from; index < limit; index += 1) {
    const instruction = instructions[index]!;
    if (instruction.op !== "jmp") continue;
    const join = indexOfOffset(instructions, branchTarget(instruction));
    // the jmp has to leave the arm it ends, and land inside the region being translated
    if (join > index && join >= target && join <= regionEnd) return { jmp: index, join };
  }
  return undefined;
}

/** Loop heads, from the branch that closes each one: the last backward branch to a head is where that loop ends. */
type LoopHeads = Map<number, number>;

function findLoops(instructions: readonly Instruction[]): LoopHeads {
  const heads: LoopHeads = new Map();
  for (let index = 0; index < instructions.length; index += 1) {
    const instruction = instructions[index]!;
    if (!instruction.op.startsWith("j")) continue;
    // a jump through a value has no displacement, so it closes no loop this can see
    if (instruction.operands[0]?.kind !== "immediate") continue;
    const target = branchTarget(instruction);
    const targetIndex = instructions.findIndex((entry) => entry.offset === target);
    if (targetIndex < 0 || targetIndex >= index) continue;
    const known = heads.get(targetIndex);
    if (known === undefined || known < index) heads.set(targetIndex, index);
  }
  return heads;
}

/**
 * A loop, from its head to the branch that closes it.
 *
 * WASM has no jump back either, so a loop is a `loop` and the backward branch is `br_if 0`, re-entering it. Leaving from
 * inside needs somewhere to go, so a loop that contains a branch to the instruction after it is wrapped in a `block`, and
 * that branch is `br_if 1`: out of the loop, out of the block, and on. A loop with no such branch needs no block, and
 * gets none, because an unused one is a nesting level that means nothing.
 */
function translateLoop(
  instructions: readonly Instruction[],
  head: number,
  bodyEnd: number,
  scope: Scope,
  out: number[],
  loops: LoopHeads,
  options: TranslateOptions,
): number {
  const exit = bodyEnd + 1;
  const exitOffset = instructions[exit]?.offset;
  const leaves = instructions
    .slice(head, bodyEnd + 1)
    .some((instruction) => instruction.op.startsWith("j") && exitOffset !== undefined && branchTarget(instruction) === exitOffset);
  if (leaves) out.push(OP.block, OP.emptyBlockType);
  out.push(OP.loop, OP.emptyBlockType);
  const nested: Scope = new Map([...scope].map(([key, value]) => [key, value + (leaves ? 2 : 1)] as const));
  nested.set(head, 0);
  if (leaves) nested.set(exit, 1);
  translateRange(instructions, head, bodyEnd + 1, nested, out, { set: false, width: 64 }, loops, options, true);
  out.push(OP.end);
  if (leaves) out.push(OP.end);
  return exit;
}

/**
 * Translates `[from, to)` into `out`, returning where it stopped.
 *
 * Branches become structure rather than jumps, because WASM has no jump: a forward conditional branch is an `if` around
 * the bytes it skips, with the condition negated, and a branch to the end of an enclosing region is `br_if` at the depth
 * that reaches it. What that costs is generality: a jump into the middle of another branch, or backwards, is refused by
 * name, since those are loops and irreducible control flow, and neither can be expressed by nesting.
 */
function translateRange(
  instructions: readonly Instruction[],
  from: number,
  to: number,
  scope: Scope,
  out: number[],
  flags: Flags,
  loops: LoopHeads,
  options: TranslateOptions,
  /** True when the caller has already opened this loop, so the head must not be opened a second time. */
  insideLoop = false,
): number {
  let index = from;
  while (index < to) {
    const instruction = instructions[index]!;
    const loopEnd = insideLoop ? undefined : loops.get(index);
    insideLoop = false;
    if (loopEnd !== undefined && loopEnd >= index && loopEnd < to) {
      index = translateLoop(instructions, index, loopEnd, scope, out, loops, options);
      flags.set = false;
      continue;
    }
    if (instruction.op === "xgetbv") {
      /*
       * `xgetbv` answers with the extended control register `%ecx` names, which is the operating system's statement of which vector
       * state it has turned on. This machine's answer is the one a processor with `x87` and SSE reports and nothing above it: 3.
       *
       * That is not a placeholder. Every AVX instruction this translator cannot decode sits behind the bit the answer leaves clear, so
       * what a program does with it is take the path this translator can run, which is the same choice a real machine without AVX
       * makes. Answering with the AVX bit set would send libc into instructions that would then refuse.
       */
      out.push(...i64Const(3n), ...globalSet(REGISTERS.indexOf("rax")), ...i64Const(0n), ...globalSet(REGISTERS.indexOf("rdx")));
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "cpuid") {
      /*
       * `cpuid` asks what this machine can do, and this machine's answer is nothing optional: the leaf in `%eax` is a question
       * about a CPU that does not exist here, so every answer is zero, which says "no vendor reported, no features beyond the
       * base leaf". A program that asks takes the path for a machine without them, which is a real x86-64 and not a fiction
       * about one. Answering with invented feature bits would send a program down a path this translator cannot run.
       */
      out.push(
        ...i64Const(0n), ...globalSet(REGISTERS.indexOf("rax")),
        ...i64Const(0n), ...globalSet(REGISTERS.indexOf("rbx")),
        ...i64Const(0n), ...globalSet(REGISTERS.indexOf("rcx")),
        ...i64Const(0n), ...globalSet(REGISTERS.indexOf("rdx")),
      );
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "hlt") {
      /*
       * `hlt` is privileged: in user mode it traps, and the bytes after it are the entry point's padding. `unreachable` is that
       * trap, and it also says the rest of the path is dead rather than leaving a value on the stack that nothing asked for.
       */
      out.push(OP.unreachable);
      index += 1;
      continue;
    }
    if (instruction.op === "nop" || instruction.op === "endbr64" || instruction.op === "prefetch") {
      // nothing observable: no register, no flag, and no memory is read even when the operand names an address, which is why a
      // nop inside a function whose image was not supplied still translates
      index += 1;
      continue;
    }
    if (instruction.op === "ret") {
      if (instruction.operands.some((operand) => operand.kind === "immediate")) {
        throw new TranslateError(instruction.offset, "ret with a stack adjustment is not translated yet");
      }
      // `return` rather than a value left on the stack: a ret inside a branch cannot leave the function's result there
      out.push(...globalGet(REGISTERS.indexOf("rax")), OP.returnOp);
      /*
       * A body can have more than one exit, and a `ret` in the middle of one is ordinary when a branch lands in a block after it:
       * `jne L; ret; L: ...`. The bytes after the `return` are unreachable along this path and reachable from elsewhere, so they
       * are translated in place. Instructions after a `return` are valid WASM, and the block a jump lands in is the guard the
       * jump itself opens around the bytes it skips.
       */
      index += 1;
      continue;
    }
    if (instruction.op === "jmp") {
      if (options.tailCalls?.has(instruction.offset) === true) {
        out.push(...translateTailCall(instruction, options));
        // a call and a return end the region as surely as the `ret` this jump replaced
        return to;
      }
      const target = indexOfOffset(instructions, branchTarget(instruction));
      const exit = scope.get(target);
      if (exit !== undefined) {
        // an unconditional exit from the region: `br` rather than `br_if`, since the jump is not conditional on anything
        out.push(OP.br, ...unsigned(exit));
        // nothing after an unconditional exit is reachable inside this region, so the region ends here
        return to;
      }
      if (target <= index) {
        throw new TranslateError(
          instruction.offset,
          "a backward jmp needs a loop, and this one does not begin where a loop is opened",
        );
      }
      // the bytes it skips are unreachable, and are emitted behind an always-false branch so that the module stays
      // structured and a jmp past code this cannot translate is refused rather than quietly skipped
      out.push(...i32Const(0), OP.ifEmpty, OP.emptyBlockType);
      translateRange(instructions, index + 1, target, new Map(scope), out, { set: false, width: 64 }, loops, options);
      out.push(OP.end);
      flags.set = false;
      index = target;
      continue;
    }
        if (instruction.op.startsWith("j")) {
      const suffix = instruction.op.slice(1);
      if (suffix === "o" || suffix === "no" || suffix === "p" || suffix === "np") {
        throw new TranslateError(
          instruction.offset,
          `j${suffix} reads the overflow or parity flag, which this does not compute`,
        );
      }
      if (!flags.set) {
        throw new TranslateError(instruction.offset, `${instruction.op} needs a cmp or test that every path reaching it goes through`);
      }
      const target = indexOfOffset(instructions, branchTarget(instruction));
      const condition = flagCondition(suffix, flags, instruction);
      const depth = scope.get(target);
      if (depth !== undefined) {
        /*
         * The target is an exit the enclosing structure already provides, so the branch is one conditional exit and the
         * bytes under it stay where they are: when the condition is false, translation continues inline. This is the
         * commonest case by far, a jump to the end of the branch it sits in, and it is also the one where moving on to
         * the target instead of continuing would silently drop the fall-through code.
         */
        out.push(...condition, OP.brIf, ...unsigned(depth));
        // a conditional branch that is not taken leaves the flags as they were, so they stay usable
        index += 1;
        continue;
      }
      if (target <= index) {
        throw new TranslateError(
          instruction.offset,
          `a backward ${instruction.op} needs a loop, and this one does not begin where a loop is opened`,
        );
      }
      const join = findJoin(instructions, index + 1, target, target, to);
      if (join !== undefined) {
        /*
         * The if/else shape. When the condition holds, control goes to the target, so the target's arm comes first in
         * the `if`; when it does not hold, the fall-through arm is the `else`. Both arms end at the join, which is the
         * `if`'s own end, so the jmp that ended the then-arm has nothing left to do.
         */
        const shared: Scope = new Map([...scope].map(([key, value]) => [key, value + 1] as const));
        shared.set(join.join, 0);
        /*
         * Both arms inherit the flags as they stand at the branch, because a `jcc` does not change them: a comparison made
         * before the branch is still there to be read in either arm. What the join then holds depends on what each arm did, so
         * each arm reports its own state and the flags survive the join only if they survived both. That is what a conditional
         * move after an if/else needs, and it is a fact about the paths rather than about the instruction alone.
         */
        const entered: Flags = { set: flags.set, width: flags.width, kind: flags.kind };
        const then: Flags = { ...entered };
        out.push(...condition, OP.ifEmpty, OP.emptyBlockType);
        translateRange(instructions, target, join.join, new Map(shared), out, then, loops, options);
        out.push(OP.elseOp);
        const otherwise: Flags = { ...entered };
        translateRange(instructions, index + 1, join.jmp, new Map(shared), out, otherwise, loops, options);
        out.push(OP.end);
        /*
         * Two arms that compared at different widths leave the join with an answer only one of them can read, and two arms that
         * learned their flags a different way leave an answer neither can read: a carry from a bit test and a carry from a borrow
         * live in the same two locals, so a join that mixes them refuses rather than picking one.
         */
        flags.set = then.set && otherwise.set && then.width === otherwise.width;
        flags.width = then.width;
        flags.kind = then.kind === otherwise.kind ? then.kind : undefined;
        if (then.kind !== otherwise.kind) flags.set = false;
        index = join.join;
        continue;
      }
      // a jump forward inside this region: an `if` around the bytes it skips, with the condition negated
      out.push(...condition, OP.i32Eqz, OP.ifEmpty, OP.emptyBlockType);
      const nested: Scope = new Map([...scope].map(([key, value]) => [key, value + 1] as const));
      nested.set(target, 0);
      translateRange(instructions, index + 1, target, nested, out, { ...flags }, loops, options);
      out.push(OP.end);
      // after the join, a comparison made on one side is not known to have happened on the other, so the flags are
      // dropped rather than assumed
      flags.set = false;
      index = target;
      continue;
    }
    if (instruction.op === "bt") {
      /*
       * `bt value, index` writes the carry and nothing else: the bit it names becomes the carry, and every other flag is left
       * undefined, which is why the flag model now records what defined it.
       *
       * The carry is expressed in the two locals a comparison already uses, as `1 - bit` against 1, because that is the pair the
       * unsigned-below condition reads: `A < B` is a borrow, and `1 - bit < 1` is exactly `bit == 1`. So `jb` after a `bt` means the
       * bit was set and `jae` means it was not, which is how a compiled program tests a flag in a word.
       */
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined) {
        throw new TranslateError(instruction.offset, "bt needs two operands");
      }
      if (destination.kind !== "reg" && destination.kind !== "memory") {
        throw new TranslateError(instruction.offset, `a bt against a ${destination.kind} is not translated`);
      }
      const width = destination.bits;
      if (width !== 32 && width !== 64) {
        throw new TranslateError(instruction.offset, `a ${width}-bit bt is not translated: only 32 and 64`);
      }
      const raw =
        destination.kind === "memory"
          ? loadFromMemory(destination, width, instruction)
          : read(destination, instruction);
      // a 32-bit operand is read from a 64-bit local, so the bits above its width are not part of the value the index walks
      const value = width === 32 ? [...raw, ...i64Const(0xffffffffn), OP.i64And] : raw;
      /*
       * An index past the operand's width wraps at that width, so the low bits of the index are what count. What that leaves out is
       * the bit-string form, where an index past the width addresses the next word along; here it wraps, which is the register form's
       * rule and not the memory form's.
       */
      const bit = [
        ...operandCode64(source, instruction),
        ...i64Const(BigInt(width - 1)),
        OP.i64And,
        OP.i64ShrU,
        ...i64Const(1n),
        OP.i64And,
      ];
      out.push(
        ...value, ...bit,
        ...localSet(CMP_A),
        ...i64Const(1n), ...localGet(CMP_A), OP.i64Xor, ...localSet(CMP_A),
        ...i64Const(1n), ...localSet(CMP_B),
      );
      flags.set = true;
      flags.width = width;
      flags.borrows = false;
      flags.kind = "bit";
      index += 1;
      continue;
    }
    if (instruction.op === "cmpxchg") {
      const translated = translateCompareExchange(instruction);
      out.push(...translated.code);
      flags.set = true;
      flags.width = translated.width;
      flags.borrows = true;
      flags.kind = undefined;
      index += 1;
      continue;
    }
    if (instruction.op === "cmp" || instruction.op === "test") {
      const translated = translateFlags(instruction);
      out.push(...translated.code);
      flags.set = true;
      flags.width = translated.width;
      flags.borrows = instruction.op === "cmp";
      // a comparison defines every condition a compiler reads, so what a shift or a bit test said about them is replaced
      flags.kind = undefined;
      index += 1;
      continue;
    }
    if (instruction.op.startsWith("cmov")) {
      const suffix = instruction.op.slice(4);
      if (suffix === "o" || suffix === "no" || suffix === "p" || suffix === "np") {
        throw new TranslateError(
          instruction.offset,
          `cmov${suffix} reads the overflow or parity flag, which this does not compute`,
        );
      }
      if (!flags.set) {
        throw new TranslateError(instruction.offset, `${instruction.op} needs a cmp or test that every path reaching it goes through`);
      }
      out.push(
        ...flagCondition(suffix, flags, instruction),
        OP.ifEmpty,
        OP.emptyBlockType,
        ...translateConditionalMove(instruction),
        OP.end,
      );
      // the instruction leaves the flags exactly as it found them, so they stay usable after it
      index += 1;
      continue;
    }
    if (instruction.op.startsWith("set") && instruction.op !== "set") {
      /*
       * `sete %al` writes the condition as a byte: one when it holds, zero when it does not. The comparison locals already hold the
       * operands the condition is about, so the byte is the condition itself, widened to the register the write expects.
       */
      const suffix = instruction.op.slice(3);
      if (suffix === "o" || suffix === "no" || suffix === "p" || suffix === "np") {
        throw new TranslateError(
          instruction.offset,
          `${instruction.op} reads the overflow or parity flag, which this does not compute`,
        );
      }
      if (!flags.set) {
        throw new TranslateError(instruction.offset, `${instruction.op} needs a cmp or test that every path reaching it goes through`);
      }
      const [operand] = instruction.operands;
      if (operand === undefined) throw new TranslateError(instruction.offset, `${instruction.op} needs a place to write`);
      const condition = flagCondition(suffix, flags, instruction);
      if (operand.kind === "memory") {
        if (operand.bits !== 8) {
          throw new TranslateError(instruction.offset, `a ${operand.bits}-bit ${instruction.op} is not an encoding: a byte is what it writes`);
        }
        out.push(...addressOf(operand, instruction), ...condition, OP.i32Store8, ...memarg(0));
      } else if (operand.kind === "reg" && operand.bits === 8) {
        out.push(...condition, WIDEN_32, ...write(operand, instruction));
      } else {
        throw new TranslateError(
          instruction.offset,
          `a ${instruction.op} into a ${operand.kind === "reg" ? `${operand.bits}-bit register` : operand.kind} is not an encoding`,
        );
      }
      // like `cmp`, the instruction leaves the flags exactly as it found them
      index += 1;
      continue;
    }
    if (instruction.op === "push" || instruction.op === "pop") {
      out.push(
        ...(instruction.op === "push"
          ? translatePush(instruction, options)
          : translatePop(instruction, options)),
      );
      // `push` and `pop` move memory and touch no flag, so a condition set before one is still readable after it
      index += 1;
      continue;
    }
    if (instruction.op === "syscall") {
      if (options.syscalls !== true) {
        throw new TranslateError(
          instruction.offset,
          "syscall needs the syscall layer: pass syscalls: true and instantiate with the machine's imports",
        );
      }
      /*
       * The call number is in `%rax` and the arguments in `%rdi`, `%rsi`, `%rdx`, `%r10`, `%r8`, `%r9`, in that order. Nothing
       * is moved, because the host reads the same registers the kernel does: the arguments are already where they belong.
       */
      out.push(
        ...globalGet(REGISTERS.indexOf("rax")),
        ...globalGet(REGISTERS.indexOf("rdi")),
        ...globalGet(REGISTERS.indexOf("rsi")),
        ...globalGet(REGISTERS.indexOf("rdx")),
        ...globalGet(REGISTERS.indexOf("r10")),
        ...globalGet(REGISTERS.indexOf("r8")),
        ...globalGet(REGISTERS.indexOf("r9")),
        OP.call, ...unsigned(SYSCALL_INDEX),
        ...globalSet(REGISTERS.indexOf("rax")),
        /*
         * The real instruction clobbers `%rcx` and `%r11`: the kernel leaves the return address in the first and `RFLAGS` in the
         * second. Zero is not the value it leaves, but leaving them alone would let a program depend on a register the kernel
         * overwrites, which is worse than an obviously wrong number.
         */
        ...i64Const(0n), ...globalSet(REGISTERS.indexOf("rcx")),
        ...i64Const(0n), ...globalSet(REGISTERS.indexOf("r11")),
      );
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "leave") {
      // `leave` moves two registers and no flags
      out.push(...translateLeave(instruction, options));
      index += 1;
      continue;
    }
    if (instruction.op === "call" && instruction.operands[0]?.kind !== "immediate") {
      /*
       * An indirect call: the target is a value the program computed, so which function runs is decided at run time and cannot be
       * an index in the module's function list. What it *can* be is one of the functions this program has translated, so the target
       * is compared against each of them and the one that matches is called. A target that matches nothing is `unreachable`, which
       * is the trap a jump to garbage is.
       *
       * The set is the functions this translation knows: the entry, and everything a direct call reached. A function whose address
       * reaches the program only as a pointer (`main`, whose address is passed to `__libc_start_main`) is not in it unless the
       * caller says so through `indirectTargets`.
       */
      const operand = instruction.operands[0];
      if (operand === undefined) throw new TranslateError(instruction.offset, "call needs a target");
      if (options.calls === undefined) {
        throw new TranslateError(
          instruction.offset,
          "an indirect call needs a program to call into: use translateProgram, which knows the functions it translated",
        );
      }
      const candidates = options.calls.candidates?.() ?? [];
      const known = [...candidates, ...(options.indirectTargets ?? [])].map((address) => ({
        address,
        index: options.calls!.indexOf(address, instruction),
      }));
      // read the target once, because every comparison below reads it again
      out.push(...(operand.kind === "memory" ? loadFromMemory(operand, 64, instruction) : read(operand, instruction)));
      out.push(...localSet(INDIRECT), OP.block, OP.emptyBlockType);
      for (const candidate of known) {
        out.push(
          ...localGet(INDIRECT), ...i64Const(candidate.address), OP.i64Eq, OP.ifEmpty, OP.emptyBlockType,
          OP.call, ...unsigned(candidate.index + (options.syscalls === true ? 1 : 0)),
          ...globalSet(REGISTERS.indexOf("rax")),
          // out of the block that holds the whole dispatch, past the comparisons that have not run
          OP.br, ...unsigned(1),
          OP.end,
        );
      }
      out.push(OP.unreachable, OP.end);
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "call") {
      // a call's displacement is relative to the byte after it, measured from this function's own start
      const displacement = instruction.operands[0];
      if (displacement === undefined || displacement.kind !== "immediate") {
        throw new TranslateError(instruction.offset, "call without a displacement is not something x86 encodes");
      }
      /*
       * Measured from the body's own first byte, through the same fallback a RIP-relative operand uses: a body translated with only
       * the program's address knows is the entry, and reading its displacement against zero would aim every call it makes at a
       * number that is a displacement and not an address.
       */
      const target = bodyBase(options) + BigInt(instruction.offset + instruction.length) + BigInt(displacement.value);
      if (options.calls === undefined) {
        throw new TranslateError(
          instruction.offset,
          "call needs a program to call into: use translateProgram, which follows the target and translates it",
        );
      }
      // the arguments are already where SysV puts them, because the machine's registers are module globals
      out.push(
        OP.call,
        ...unsigned(options.calls.indexOf(target, instruction) + (options.syscalls === true ? 1 : 0)),
        ...globalSet(REGISTERS.indexOf("rax")),
      );
      // a callee may clobber every flag and every caller-saved register, so nothing about the condition survives
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "lea") {
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined) {
        throw new TranslateError(instruction.offset, "lea needs a destination and an address");
      }
      if (source.kind !== "memory" || source.addressCode === undefined) {
        throw new TranslateError(instruction.offset, "lea needs a memory operand and a resolved address");
      }
      // `lea` computes an address and writes a register, and the flags it finds are the flags it leaves
      out.push(...source.addressCode, ...write(destination, instruction));
      index += 1;
      continue;
    }
    if (instruction.op === "cwde" || instruction.op === "cdqe" || instruction.op === "cdq" || instruction.op === "cqo") {
      /*
       * The four instructions that widen a value's sign, which a division needs before it can read `%rdx:%rax` as one signed
       * number. `cwde` and `cdqe` write `%rax` from its own low 32 bits, and `cdq` and `cqo` write `%rdx` as all ones or all
       * zeroes from the sign of `%rax`.
       */
      const rax = REGISTERS.indexOf("rax");
      if (instruction.op === "cwde" || instruction.op === "cdqe") {
        // the low 32 bits shifted up and back down arithmetically, which is the sign extension itself
        out.push(...globalGet(rax), ...i64Const(32n), OP.i64Shl, ...i64Const(32n), OP.i64ShrS, ...globalSet(rax));
      } else {
        const shift = instruction.op === "cqo" ? 63n : 31n;
        out.push(...globalGet(rax), ...i64Const(shift), OP.i64ShrS, ...globalSet(REGISTERS.indexOf("rdx")));
      }
      index += 1;
      continue;
    }
    if (instruction.op === "movsxd") {
      /*
       * `movsxd` loads 32 bits and keeps their sign, which is how a `switch` table's entry is read: an offset from the table, which may
       * be negative, widened to the width the table's own address is added at. The load itself is the same as any 32-bit read, so this
       * is the load, a wrap to its width, and the sign extension the instruction is named for.
       */
      const [destination, source] = instruction.operands;
      if (destination?.kind !== "reg" || source === undefined || source.kind === "immediate") {
        throw new TranslateError(instruction.offset, "movsxd needs a register to write and a 32-bit value to read");
      }
      const loaded = source.kind === "memory" ? loadFromMemory(source, 32, instruction) : read(source, instruction);
      out.push(...loaded, NARROW_32, OP.i64ExtendI32S, ...write(destination, instruction));
      index += 1;
      continue;
    }
    if (instruction.op === "movzx" || instruction.op === "movsx") {
      /*
       * `movzx` and `movsx` widen an 8- or 16-bit source into a 32- or 64-bit register: the zero form keeps only the source's bits,
       * the sign form keeps its sign. That is how a compiled program reads a byte out of a structure, so the byte load has to carry
       * its own sign rather than borrowing the sign-extending form every other byte operand uses.
       */
      const [destination, source] = instruction.operands;
      if (destination?.kind !== "reg" || (destination.bits !== 32 && destination.bits !== 64)) {
        throw new TranslateError(
          instruction.offset,
          `a ${instruction.op} into a ${destination?.kind ?? "missing"} operand is not an encoding: the destination is a 32- or 64-bit register`,
        );
      }
      const sourceBits = source === undefined || source.kind === "immediate" ? undefined : source.bits;
      if (source === undefined || source.kind === "immediate" || (sourceBits !== 8 && sourceBits !== 16)) {
        throw new TranslateError(
          instruction.offset,
          `a ${instruction.op} from a ${sourceBits ?? "missing"} operand is not an encoding: the source is a byte or a word`,
        );
      }
      const sign = instruction.op === "movsx";
      const shift = 64n - BigInt(sourceBits);
      const signed = (code: number[]): number[] => [...code, ...i64Const(shift), OP.i64Shl, ...i64Const(shift), OP.i64ShrS];
      let loaded: number[];
      if (source.kind === "memory") {
        const load = source.bits === 8 ? (sign ? OP.i32Load8S : OP.i32Load8U) : sign ? OP.i32Load16S : OP.i32Load16U;
        const widen = sign ? OP.i64ExtendI32S : WIDEN_32;
        loaded = [...addressOf(source, instruction), load, ...memarg(0), widen];
      } else {
        // a register read is already masked to the byte or the word, so only the sign still has to be put back
        loaded = read(source, instruction);
        if (sign) loaded = signed(loaded);
      }
      out.push(...loaded, ...write(destination, instruction));
      index += 1;
      continue;
    }
    if (instruction.op === "mul" || instruction.op === "imul1" || instruction.op === "div" || instruction.op === "idiv") {
      /*
       * The four instructions that read and write registers the encoding does not name. `%rax` is the accumulator and `%rdx` the
       * high half of a product or the remainder, so the translation loads them into locals, does the arithmetic, and writes both
       * back: nothing else in the instruction list can observe a half-written register file.
       */
      const [source] = instruction.operands;
      if (source === undefined) throw new TranslateError(instruction.offset, `${instruction.op} needs an operand`);
      if (source.kind === "reg" && source.bits !== 64) {
        throw new TranslateError(
          instruction.offset,
          `a ${source.bits}-bit ${instruction.op} is not translated: its dividend would be %edx:%eax, which is not modelled`,
        );
      }
      out.push(
        ...(source.kind === "memory" ? loadFromMemory(source, 64, instruction) : read(source, instruction)),
        ...localSet(DIV_DIVISOR),
        ...globalGet(REGISTERS.indexOf("rax")), ...localSet(DIV_LO),
        ...globalGet(REGISTERS.indexOf("rdx")), ...localSet(DIV_HI),
      );
      if (instruction.op === "mul") {
        // `%rdx:%rax = %rax * source`, unsigned: the product is wider than either operand
        out.push(
          ...multiplyWide(DIV_LO, DIV_DIVISOR),
          ...localGet(DIV_Q), ...globalSet(REGISTERS.indexOf("rax")),
          ...localGet(DIV_SIGN_HI), ...globalSet(REGISTERS.indexOf("rdx")),
        );
      } else if (instruction.op === "imul1") {
        // the same, on magnitudes, with the product negated when the operands' signs differed
        out.push(
          ...localGet(DIV_DIVISOR), ...i64Const(0n), OP.i64LtS, OP.i64ExtendI32U, ...localSet(DIV_FLAG),
          ...localGet(DIV_LO), ...i64Const(0n), OP.i64LtS, OP.i64ExtendI32U, ...localSet(DIV_FLAG2),
          ...negateIf(DIV_DIVISOR, [...localGet(DIV_FLAG)]),
          ...negateIf(DIV_LO, [...localGet(DIV_FLAG2)]),
          ...localGet(DIV_FLAG), ...localGet(DIV_FLAG2), OP.i64Xor, ...localSet(DIV_FLAG),
          ...multiplyWide(DIV_LO, DIV_DIVISOR),
          ...negateWideIf(DIV_SIGN_HI, DIV_Q, [...localGet(DIV_FLAG)]),
          ...localGet(DIV_Q), ...globalSet(REGISTERS.indexOf("rax")),
          ...localGet(DIV_SIGN_HI), ...globalSet(REGISTERS.indexOf("rdx")),
        );
      } else if (instruction.op === "div") {
        // a zero divisor is #DE rather than a quotient, and the trap happens before anything is written
        out.push(
          ...trapUnless([...localGet(DIV_DIVISOR), OP.i64Eqz]),
          ...divideUnsigned(),
          ...localGet(DIV_Q), ...globalSet(REGISTERS.indexOf("rax")),
          ...localGet(DIV_REM), ...globalSet(REGISTERS.indexOf("rdx")),
        );
      } else {
        /*
         * `idiv`: both operands are negated to magnitudes first, because the long division is unsigned. The dividend's sign is the
         * top bit of `%rdx`, the quotient's is the two signs differing, and the remainder's is the dividend's own, since the
         * processor rounds the quotient toward zero.
         */
        out.push(
          ...trapUnless([...localGet(DIV_DIVISOR), OP.i64Eqz]),
          ...localGet(DIV_DIVISOR), ...i64Const(0n), OP.i64LtS, OP.i64ExtendI32U, ...localSet(DIV_FLAG),
          ...localGet(DIV_HI), ...i64Const(0n), OP.i64LtS, OP.i64ExtendI32U, ...localSet(DIV_FLAG2),
          ...localGet(DIV_FLAG), ...localGet(DIV_FLAG2), OP.i64Xor, ...localSet(DIV_SIGN_Q),
          ...negateIf(DIV_DIVISOR, [...localGet(DIV_FLAG)]),
          ...negateWideIf(DIV_HI, DIV_LO, [...localGet(DIV_FLAG2)]),
          ...divideUnsigned(),
          ...negateIf(DIV_Q, [...localGet(DIV_SIGN_Q)]),
          ...negateIf(DIV_REM, [...localGet(DIV_FLAG2)]),
          ...localGet(DIV_Q), ...globalSet(REGISTERS.indexOf("rax")),
          ...localGet(DIV_REM), ...globalSet(REGISTERS.indexOf("rdx")),
        );
      }
      // `mul` and `imul` define the carry and overflow flags from the high half, and neither flag is computed here
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "pand" || instruction.op === "pandn" || instruction.op === "por" || instruction.op === "pxor") {
      /*
       * Bitwise operations on 128 bits, which are two 64-bit operations here because a half is a register. The source's half is read
       * before the destination's is written, so `pand %xmm0,%xmm0` is the identity it is on the processor rather than a read of a
       * half-written register.
       */
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined || destination.kind !== "reg" || source.kind !== "reg") {
        throw new TranslateError(instruction.offset, `${instruction.op} takes two vector registers`);
      }
      const combine = { pand: OP.i64And, pandn: OP.i64And, por: OP.i64Or, pxor: OP.i64Xor }[instruction.op] as number;
      for (const [target, from] of [
        [destination, source],
        [otherHalf(destination, instruction), otherHalf(source, instruction)],
      ] as const) {
        // `pandn` is `~destination & source`, so its destination is inverted first
        out.push(...read(from, instruction), ...localSet(SSE_A));
        if (instruction.op === "pandn") {
          out.push(...read(target, instruction), ...i64Const(-1n), OP.i64Xor, ...localGet(SSE_A), combine, ...write(target, instruction));
        } else {
          out.push(...read(target, instruction), ...localGet(SSE_A), combine, ...write(target, instruction));
        }
      }
      index += 1;
      continue;
    }
    if (
      instruction.op === "movaps" ||
      instruction.op === "movups" ||
      instruction.op === "movdqa" ||
      instruction.op === "movdqu"
    ) {
      /*
       * 128 bits between a vector register and memory. WASM has no 16-byte load or store, so this is two 64-bit accesses at the same
       * address, eight bytes apart: the low half at the operand's own offset and the high half at the memarg's next one.
       */
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined) {
        throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
      }
      if (destination.kind === "memory" && source.kind === "reg" && isVectorHalf(source)) {
        out.push(
          ...addressOf(destination, instruction), ...read(source, instruction), OP.i64Store, ...memarg(0),
          ...addressOf(destination, instruction), ...read(otherHalf(source, instruction), instruction), OP.i64Store, ...memarg(8),
        );
        index += 1;
        continue;
      }
      if (destination.kind === "reg" && isVectorHalf(destination) && source.kind === "memory") {
        out.push(
          ...addressOf(source, instruction), OP.i64Load, ...memarg(0), ...write(destination, instruction),
          ...addressOf(source, instruction), OP.i64Load, ...memarg(8), ...write(otherHalf(destination, instruction), instruction),
        );
        index += 1;
        continue;
      }
      throw new TranslateError(instruction.offset, `${instruction.op} moves 128 bits between memory and a vector register`);
    }
    if (instruction.op === "movlps" || instruction.op === "movhps") {
      /*
       * A 64-bit move into or out of one named half of a vector register. The half that is not named keeps what it had, which is the
       * difference between these and `movq`: `movq` into a vector register clears the half above the value it writes.
       */
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined) {
        throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
      }
      if (destination.kind === "memory" && source.kind === "reg" && isVectorHalf(source)) {
        out.push(...translateMovToMemory(destination, source, instruction));
        index += 1;
        continue;
      }
      if (destination.kind === "reg" && isVectorHalf(destination) && source.kind === "memory") {
        out.push(...loadFromMemory(source, 64, instruction), ...write(destination, instruction));
        index += 1;
        continue;
      }
      throw new TranslateError(instruction.offset, `${instruction.op} moves 64 bits between memory and one half of a vector register`);
    }
    if (instruction.op === "movd" || instruction.op === "movq") {
      /*
       * The two directions of a vector move are different instructions: into a vector register the whole 128 bits are written,
       * because `movd` clears the upper half, and out of one it is an ordinary store of the low 64 bits. The second is what the
       * `mov`-to-memory path already does, given a register operand that holds those 64 bits, which is how the vector file is modelled.
       */
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined) {
        throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
      }
      if (instruction.op === "movq" && destination.kind === "reg" && isVectorHalf(destination)) {
        // into a vector register from memory or from another one: the upper half is cleared, as `movd` clears everything above
        const value = source.kind === "memory" ? loadFromMemory(source, 64, instruction) : read(source, instruction);
        out.push(
          ...value,
          ...write(destination, instruction),
          ...i64Const(0n),
          ...globalSet(slotOf(otherHalf(destination, instruction), instruction).local),
        );
        index += 1;
        continue;
      }
      if (instruction.op === "movq") {
        if (destination.kind !== "memory" || source.kind !== "reg") {
          throw new TranslateError(instruction.offset, "movq from a vector register stores its low 64 bits, so it needs memory");
        }
        out.push(...translateMovToMemory(destination, source, instruction));
        index += 1;
        continue;
      }
      if (destination.kind === "reg" && isVectorHalf(destination)) {
        const value =
          source.kind === "memory"
            ? loadFromMemory(source, source.bits === 64 ? 64 : 32, instruction)
            : read(source, instruction);
        const zeroExtend = destination.bits === 32 ? [NARROW_32, WIDEN_32] : [];
        out.push(
          ...value,
          ...zeroExtend,
          ...write(destination, instruction),
          // the upper half is cleared, which is what makes `movd` a write of the whole register rather than of its low half
          ...i64Const(0n),
          ...globalSet(slotOf(otherHalf(destination, instruction), instruction).local),
        );
        index += 1;
        continue;
      }
      if (source.kind === "reg" && isVectorHalf(source)) {
        // out of a vector register: four bytes into a register or into memory, which is an ordinary `mov` at that width
        out.push(...translateMov({ ...instruction, op: "mov", operands: [destination, source] }));
        index += 1;
        continue;
      }
      throw new TranslateError(instruction.offset, "movd needs a vector register on exactly one side");
    }
    if (instruction.op === "punpckldq") {
      /*
       * `punpckldq %xmm2, %xmm0` interleaves the four low dwords: the result is `[dst0, src0, dst1, src1]`, so each half of the
       * destination is a dword of its own half or'ed with a dword of the source's, shifted up. The two halves are read into
       * scratch locals first, because the destination's low half is written after its high half has been computed from it.
       */
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined || destination.kind !== "reg" || source.kind !== "reg") {
        throw new TranslateError(instruction.offset, "punpckldq takes two vector registers");
      }
      out.push(...read(destination, instruction), ...localSet(SSE_A));
      out.push(...read(source, instruction), ...localSet(SSE_B));
      const dword = (local: number, shift: bigint): number[] => [
        ...localGet(local),
        ...(shift === 0n ? [] : [...i64Const(shift), OP.i64ShrU]),
        ...i64Const(0xffffffffn),
        OP.i64And,
      ];
      out.push(
        ...dword(SSE_A, 32n),
        ...dword(SSE_B, 32n),
        ...i64Const(32n),
        OP.i64Shl,
        OP.i64Or,
        ...globalSet(slotOf(otherHalf(destination, instruction), instruction).local),
        ...dword(SSE_A, 0n),
        ...dword(SSE_B, 0n),
        ...i64Const(32n),
        OP.i64Shl,
        OP.i64Or,
        ...globalSet(slotOf(destination, instruction).local),
      );
      index += 1;
      continue;
    }
    if (instruction.op === "punpcklqdq") {
      /*
       * `punpcklqdq %xmm1, %xmm0` interleaves the low quadwords, and since a quadword is exactly a half of the register, that is a
       * pair of moves: the destination's low half goes up into its high half, and the source's low half becomes the low half.
       * The destination's old low half is read first, because writing the low half is what overwrites it.
       */
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined || destination.kind !== "reg" || source.kind !== "reg") {
        throw new TranslateError(instruction.offset, "punpcklqdq takes two vector registers");
      }
      out.push(...read(destination, instruction), ...localSet(SSE_A));
      out.push(...read(source, instruction), ...write(destination, instruction));
      out.push(...localGet(SSE_A), ...write(otherHalf(destination, instruction), instruction));
      index += 1;
      continue;
    }
    if (instruction.op === "mov") {
      out.push(...translateMov(instruction));
      index += 1;
      continue;
    }
    if (instruction.op === "not" || instruction.op === "neg") {
      const translated = translateUnary(instruction);
      out.push(...translated.code);
      if (translated.width !== undefined) {
        flags.set = true;
        flags.width = translated.width;
        // `neg` is zero minus a value, so its borrow is a carry; `not` sets no flag at all and has no width
        flags.borrows = instruction.op === "neg";
      }
      index += 1;
      continue;
    }
    if (instruction.op === "sbb") {
      /*
       * A subtraction that also subtracts the carry, which is how a compiler turns a comparison into a number: `sbb %rbx,%rbx` is
       * zero minus the borrow, so `%rbx` becomes all ones when the comparison borrowed and zero when it did not.
       *
       * The borrow is read out of the comparison the flags still hold: `a - b` borrows exactly when `a < b` as unsigned values. After
       * a `test` there is no borrow at all, because a test subtracts nothing.
       */
      if (!flags.set || flags.borrows === undefined) {
        throw new TranslateError(
          instruction.offset,
          "sbb reads the carry flag, which needs a cmp or test that every path reaching it goes through",
        );
      }
      if (flags.kind === "shift" || flags.kind === "zero") {
        throw new TranslateError(
          instruction.offset,
          `sbb after a ${flags.kind === "zero" ? "bit scan" : "shift"} reads the carry, which it does not define: a shift's carry is ` +
            "the bit that left and a scan's is not defined at all, so neither is a borrow to subtract here",
        );
      }
      const [destination, source] = instruction.operands;
      if (destination === undefined || source === undefined) {
        throw new TranslateError(instruction.offset, "sbb needs two operands");
      }
      const carry =
        flags.borrows === false
          ? [...i64Const(0n)]
          : flags.width === 32
            ? [...comparisonOperands(32), OP.i32LtU, OP.i64ExtendI32U]
            : flags.width === 16
              ? [...comparisonOperands(16), OP.i64LtU, OP.i64ExtendI32U]
              : [...comparisonOperands(64), OP.i64LtU, OP.i64ExtendI32U];
      out.push(
        ...carry,
        ...localSet(SBB_CARRY),
        ...read(destination, instruction),
        ...read(source, instruction),
        OP.i64Sub,
        ...localGet(SBB_CARRY),
        OP.i64Sub,
        ...write(destination, instruction),
      );
      // the arithmetic flags it sets are not modelled, so the comparison it inherited is spent
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "bsf" || instruction.op === "bsr") {
      /*
       * Its flags are about the source, not about the index it wrote: the zero flag says whether a bit was found at all, and every
       * other flag is undefined. `kind: "zero"` is how a condition is told so.
       */
      out.push(...translateBitScan(instruction));
      flags.set = true;
      const destination = instruction.operands[0];
      flags.width = destination !== undefined && destination.kind === "reg" && destination.bits === 32 ? 32 : 64;
      flags.kind = "zero";
      index += 1;
      continue;
    }
    if (instruction.op === "shl" || instruction.op === "shr" || instruction.op === "sar") {
      /*
       * A shift's flags describe its result the way an `or`'s do, and its carry describes a bit that has left, which nothing here
       * computes: `kind: "shift"` is how a later condition is told which of the two it is asking about.
       */
      out.push(...translateShift(instruction));
      flags.set = true;
      const destination = instruction.operands[0];
      flags.width = destination !== undefined && destination.kind === "reg" && destination.bits === 32 ? 32 : 64;
      flags.kind = "shift";
      index += 1;
      continue;
    }
    if (instruction.op in BINARY) {
      out.push(...translateBinary(instruction));
      if (FLAG_SETTING.has(instruction.op)) {
        // an arithmetic result sets the flags a branch can read, the whole set of them rather than the carry alone
        flags.set = true;
        flags.kind = undefined;
        const destination = instruction.operands[0];
        flags.width = destination !== undefined && destination.kind === "reg" && destination.bits === 32 ? 32 : 64;
      } else {
        // a multiplication sets the flags too, to values this does not compute, so what was known before it is gone
        flags.set = false;
      }
      index += 1;
      continue;
    }
    throw new TranslateError(instruction.offset, `${instruction.op} is not translated yet`);
  }
  return index;
}

/** One translated function, with the address it came from so a call can be recognised as reaching it. */
interface Body {
  readonly address: bigint;
  readonly instructions: readonly Instruction[];
  /** The jumps in this body that leave its bytes, by offset, and where each one goes. */
  readonly tailCalls: ReadonlyMap<number, bigint>;
  /** True when the address does not name code, so the body is a trap rather than a translation of any bytes. */
  readonly traps?: boolean;
}

/**
 * The instructions of the function that starts at the beginning of `bytes`, found by walking what it can reach.
 *
 * A compiled function rarely has one `ret`: an early return or a shared epilogue gives it several, and the bytes after the
 * first one are still part of it. Cutting there would drop the epilogue, and a branch into the dropped region would be
 * reported as a bad branch target, which is true but points at the wrong thing.
 *
 * So the extent comes from reachability, and only the instructions that are walked get decoded. That matters: the bytes after
 * a function are the next function, or the zero padding the linker leaves, and a strict decode of those would fail on bytes
 * that were never instructions. What is left inside the range but unreachable is refused rather than included, because it
 * means this is not where a function starts.
 */
function functionExtent(
  bytes: Uint8Array,
  address: bigint,
  options: TranslateOptions = {},
): { readonly instructions: readonly Instruction[]; readonly tailCalls: ReadonlyMap<number, bigint> } {
  const decoded = new Map<number, Instruction>();
  // which instruction fell into which, which is how the three that build a jump table's address are found
  const previous = new Map<number, number>();
  const decodeAt = (offset: number): Instruction => {
    const known = decoded.get(offset);
    if (known !== undefined) return known;
    /*
     * A negative offset is a branch to a byte before the function's start, and the window the walk was given begins at the start,
     * so reading it would look like the buffer ending rather than what it is: the function this walk began at is not a function,
     * or the call that reached it was aimed into the middle of another body.
     */
    if (offset < 0) {
      throw new TranslateError(
        offset,
        `a branch reaches ${hex(address + BigInt(offset))}, which is before the function at ${hex(address)}: either the body ends ` +
          "in a tail call into an earlier function, which is not translated yet, or the call that reached it was aimed into the " +
          "middle of another body",
      );
    }
    if (offset >= bytes.length) {
      throw new TranslateError(
        offset,
        `the function at ${hex(address)} runs off the end of the bytes given, so its extent is unknown`,
      );
    }
    const instruction = decodeInstruction(bytes, offset);
    decoded.set(offset, instruction);
    return instruction;
  };

  const reachable = new Set<number>();
  // jumps out of the body, by offset, holding the address each one reaches
  const tailCalls = new Map<number, bigint>();
  // the offset of the first jump through a register whose table named no case, which is what a hole in the body is evidence of
  let unreadableTable: number | undefined;
  const work: number[] = [0];
  while (work.length > 0) {
    const offset = work.pop()!;
    if (reachable.has(offset)) continue;
    const instruction = decodeAt(offset);
    reachable.add(offset);
    // a ret has no successor, which is what stops the walk before the next function's bytes
    if (instruction.op === "ret") continue;
    if (instruction.op === "jmp") {
      const operand = instruction.operands[0];
      if (operand?.kind === "immediate") {
        const target = branchTarget(instruction);
        if (target < 0) {
          /*
           * A jump to a byte before this body's first one. The window the walk was given begins at the body, so the target is not
           * in it, and the function this body reached is one that starts earlier: a tail call. It is recorded and not followed,
           * because its bytes belong to another function and walking them here would make this body claim them.
           */
          tailCalls.set(offset, address + BigInt(target));
          continue;
        }
        if (!previous.has(target)) previous.set(target, offset);
        work.push(target);
        continue;
      }
      /*
       * A computed jump: the target is a value, so the walk cannot follow it. What it can do is continue where the instruction's
       * bytes stop, which is where a compiler puts the cases of the switch such a jump usually ends, and stop where there is no code
       * to continue into: a tail call through a register has the linker's padding after it, and following that would decode zeros.
       */
      /*
       * A `switch` says where its cases are in a table rather than in its bytes, so the table is read here and each entry becomes a
       * place the walk continues from. The three instructions before the jump are what say which table: they are the chain of
       * fall-throughs that led here, and they are the `lea`, the `movsxd` and the `add` that compute an entry's address.
       */
      /*
       * The chain is the run of instructions that led here, and how far back it has to go is not fixed: gcc loads the table's base
       * once, where the value it holds is still cheap, and the `cmp`, the bounds check and whatever else belongs in between. The reader
       * wants the last write to the base register, so the walk hands it the whole run and lets that be decided there.
       */
      const chain: Instruction[] = [];
      for (let cursor = offset; chain.length < 64; ) {
        const before = previous.get(cursor);
        if (before === undefined) break;
        chain.unshift(decodeAt(before));
        cursor = before;
      }
  const cases = jumpTableTargets([...chain, instruction], chain.length, address, bytes.length, options);
      // a table that named nothing is a table this could not read, which is what a hole later in the body is evidence of
      if (cases.length === 0 && unreadableTable === undefined) unreadableTable = offset;
      for (const target of cases) work.push(target);
      const afterJump = offset + instruction.length;
      if (afterJump >= bytes.length || bytes[afterJump] === 0) continue;
      try {
        decodeAt(afterJump);
      } catch {
        continue;
      }
      previous.set(afterJump, offset);
      work.push(afterJump);
      continue;
    }
    if (instruction.op.startsWith("j")) {
      /*
       * A branch target gets a predecessor of its own, so the run that reached an instruction can be read back through a jump and not
       * only through fall-through. A target that already has one keeps it: the first path the walk took is the one that reading was
       * written against.
       */
      const target = branchTarget(instruction);
      if (!previous.has(target)) previous.set(target, offset);
      work.push(target);
    }
    /*
     * A call returns to the instruction after it, and the function it names is translated separately. But a call to one that
     * never returns has no code after it, and gcc emits exactly that shape: `call __stack_chk_fail` at the end of a
     * stack-protected function, followed by the linker's zero padding. The two cases are distinguishable: a returning call
     * must be followed by an instruction, so a byte that cannot start one is not a return address. Off the end of the bytes
     * is still truncation, and still refused, because there is nothing there to distinguish it from a cut-off function.
     */
    const next = offset + instruction.length;
    if (instruction.op === "call" && next < bytes.length) {
      /*
       * A zero byte cannot start an instruction a compiler would emit (`add %al,(%rax)` is not something a call is followed
       * by), and a run of them is the linker's padding, so a call followed by one is a call to something that never returns.
       * This is tested before decodability because a zero byte *does* decode: relying on the decode to fail made this rule
       * depend on which instructions the subset happened to have, which is not a property of the program.
       */
      if (bytes[next] === 0) continue;
      try {
        decodeAt(next);
      } catch {
        continue;
      }
    }
    previous.set(next, offset);
    work.push(next);
  }

  /*
   * No reachable instruction may start inside another one. A call into the middle of an instruction reaches bytes that decode,
   * so the walk accepts them, and the function then computes something the source never said. That is the difference between an
   * address that is a function's start and one that merely decodes, and it is worth naming rather than translating.
   */
  const starts = [...reachable].sort((left, right) => left - right);
  let covered = 0;
  for (const start of starts) {
    if (start < covered) {
      throw new TranslateError(
        start,
        `${hex(address + BigInt(start))} is inside the instruction that ends at ${hex(address + BigInt(covered))}, so it is ` +
          "not where a function starts",
      );
    }
    covered = start + decodeAt(start).length;
  }

  /*
   * A call whose target lands inside one of this function's own instructions is a call into the middle of it, which no compiler
   * emits and which translates to something the source never said.
   *
   * What it is not is a call to a byte the walk never reached. The walk follows fall-through, branches and calls, so a body with a
   * branch into a cold block leaves a hole where that block's neighbours are, and the next function can start inside the hole: a
   * call to it names a real function start and not the middle of anything. The test is therefore whether an instruction covers the
   * target, not whether the target is below the far end of the walk.
   */
  const extent = starts
    .map((start) => ({ start, end: start + decodeAt(start).length }))
    .sort((left, right) => left.start - right.start);
  for (const start of starts) {
    const instruction = decodeAt(start);
    if (instruction.op !== "call") continue;
    // a call through a register has no target in the bytes to be in the middle of an instruction
    if (instruction.operands[0]?.kind !== "immediate") continue;
    const target = branchTarget(instruction);
    if (target < 0 || reachable.has(target)) continue;
    const covering = extent.find((entry) => entry.start < target && target < entry.end);
    if (covering === undefined) continue;
    throw new TranslateError(
      start,
      `the call at ${hex(address + BigInt(start))} targets ${hex(address + BigInt(target))}, which is inside the instruction at ` +
        `${hex(address + BigInt(covering.start))} and not at its start, so it is not where a function starts`,
    );
  }

  /*
   * The body is the instructions its entry reaches, and nothing else. It used to be the whole byte range from the entry to the
   * furthest byte reached, with every byte inside it required to be reached or to be `nop` padding, and that stopped being true the
   * first time a body was walked whose branches reach a cold block a long way off: the bytes in between belong to other functions,
   * and a function is not the text it sits in the middle of.
   *
   * So a hole is skipped. Its bytes are not part of this body and are not translated, and a branch that names one of them is refused
   * where the blocks are built, because an instruction this body does not hold is not an instruction it can go to. Padding stays,
   * as the dead code it is: an assembler pads a block out to a boundary, and a `nop` that nothing reaches translates to nothing.
   */
  const ordered: Instruction[] = [];
  let skipped: number | undefined;
  let offset = 0;
  for (const start of [...reachable].sort((left, right) => left - right)) {
    while (offset < start) {
      let padding: Instruction | undefined;
      try {
        padding = decodeAt(offset);
      } catch {
        padding = undefined;
      }
      if (padding === undefined || padding.op !== "nop" || offset + padding.length > start) break;
      ordered.push(padding);
      offset += padding.length;
    }
    if (offset < start && skipped === undefined) skipped = offset;
    if (offset > start) {
      throw new TranslateError(
        start,
        `${hex(address + BigInt(start))} is inside the instruction that ends at ${hex(address + BigInt(offset))}, so it is not ` +
          "where an instruction starts",
      );
    }
    const instruction = decodeAt(start);
    ordered.push(instruction);
    offset = start + instruction.length;
  }
  /*
   * A jump through a register goes somewhere its bytes do not name, so the walk finds its destinations in a table. When the table is
   * not readable the walk still reaches what falls after the jump, and the case bodies it missed are now holes: that is the one shape
   * where a hole is evidence, because the body is telling us it goes places this walk could not enumerate. It is refused rather than
   * translated into a dispatcher that would have no case to land in.
   */
  if (unreadableTable !== undefined && skipped !== undefined) {
    throw new TranslateError(
      skipped,
      `nothing reaches the bytes at 0x${skipped.toString(16)} inside the function at ${hex(address)}, and the jump through a ` +
        `register at ${hex(address + BigInt(unreadableTable))} is the reason they were expected to be reached: its table ` +
        "cannot be read, so the cases it names are unknown",
    );
  }
  return { instructions: ordered, tailCalls };
}

/** True when a branch in the list goes to itself or backwards, which no nesting of blocks can express. */
function needsDispatcher(instructions: readonly Instruction[], options: TranslateOptions = {}): boolean {
  return instructions.some((instruction, index) => {
    if (!instruction.op.startsWith("j")) return false;
    // a jump whose target is a value lands in a block, and which one is decided by that value
    if (instruction.operands[0]?.kind !== "immediate") return true;
    // a jump out of the body is a call and a return, which nests like a `ret` rather than like a branch
    if (options?.tailCalls?.has(instruction.offset) === true) return false;
    const target = indexOfOffset(instructions, branchTarget(instruction));
    return target >= 0 && target <= index;
  });
}

/**
 * Where each block of straight-line code starts: the body's first instruction, every branch target, and the instruction after a
 * branch or a `ret`, since control can arrive there without the bytes before it having run.
 */
/**
 * Which of the sixteen registers a name refers to, or undefined for one this does not map.
 *
 * `%ecx` and `%rcx` are one register written at two widths, which is exactly the pair a jump table's base is loaded into and then read
 * through, so a check that compared names would see two registers where the program has one.
 */
function registerNumberOf(name: string): number | undefined {
  const widened: Record<string, string> = {
    eax: "rax", ecx: "rcx", edx: "rdx", ebx: "rbx", esp: "rsp", ebp: "rbp", esi: "rsi", edi: "rdi",
  };
  const base = widened[name] ?? name.replace(/^r(\d+)[bwd]?$/, "r$1");
  if (base.startsWith("xmm")) return undefined;
  const index = REGISTERS.indexOf(base as (typeof REGISTERS)[number]);
  return index < 0 ? undefined : index;
}

/**
 * The targets a jump table names, read out of the image.
 *
 * A `switch` over a dense range of values compiles to a table of 32-bit offsets and four instructions that use it, in a fixed shape:
 *
 *   lea    table(%rip),%rcx       the table's own address, which is what its entries are measured from
 *   movsxd (%rcx,%rdx,4),%rdx     one entry, sign-extended: an offset rather than an address
 *   add    %rcx,%rdx              the offset plus the table, which is where the case starts
 *   jmp    *%rdx
 *
 * Reading the table is what lets the walk reach a case body no branch names, and what lets the dispatcher land in one.
 *
 * How many entries there are is not in the table. The bounds check before it compares against a case count that is already a value in a
 * register, so this reads until an entry stops naming a place inside the body: an entry that is not an offset into these bytes ends the
 * table, which is what the section's other data looks like when read as offsets. A table whose data happens to look like offsets would
 * be read too far, and the extra targets are blocks that nothing runs, so the mistake costs size rather than correctness.
 */
/** The operations whose first operand is the register they write, which is what makes one instruction the definition of another. */
const WRITING_OPERATIONS = new Set([
  "mov", "movsxd", "movzx", "movsx", "lea", "add", "sub", "and", "or", "xor", "adc", "sbb", "shl", "shr", "sar", "imul", "inc",
  "dec", "neg", "not", "pop", "xchg",
]);

/** True when this instruction writes the given register as its first operand. */
function writesRegister(instruction: Instruction, register: number): boolean {
  const [destination] = instruction.operands;
  if (destination?.kind !== "reg") return false;
  if (registerNumberOf(destination.name) !== register) return false;
  if (instruction.op.startsWith("cmov") || instruction.op.startsWith("set")) return true;
  return WRITING_OPERATIONS.has(instruction.op);
}

/**
 * The instruction that last wrote a register before the one at `index`, when the bytes between them are a straight run.
 *
 * Undefined means the register's value there is not something these bytes say. A call clobbers every caller-saved register, and a
 * `ret` means nothing after it is reached by falling through. A backwards branch is the loop that a `switch` usually sits inside, and
 * it is harmless as long as it lands at or before the write: the path from there passes the write again. A branch that lands after it
 * would reach the load with whatever the register held before, which is not a table this can read.
 *
 * A forward branch is not a problem at all: the instructions it skips are inside the span being scanned, so a write among them is
 * still found, and one that lands past the load does not reach it.
 */
function definingLea(instructions: readonly Instruction[], index: number, register: number): Instruction | undefined {
  const backwards: number[] = [];
  for (let back = index - 1; back >= 0 && index - back <= 64; back -= 1) {
    const candidate = instructions[back]!;
    if (candidate.op === "call" || candidate.op === "syscall" || candidate.op === "ret" || candidate.op === "hlt") {
      return undefined;
    }
    if (candidate.op.startsWith("j")) {
      // a jump whose target is a value is a path this cannot follow, so the register's value here is not known
      if (candidate.operands[0]?.kind !== "immediate") return undefined;
      const target = branchTarget(candidate);
      if (target < candidate.offset) backwards.push(target);
      continue;
    }
    if (writesRegister(candidate, register)) {
      return backwards.every((target) => target <= candidate.offset) ? candidate : undefined;
    }
  }
  return undefined;
}

function jumpTableTargets(
  instructions: readonly Instruction[],
  index: number,
  bodyAddress: bigint,
  bodyBytes: number,
  options: TranslateOptions,
): number[] {
  const read = options.readData;
  if (read === undefined) return [];
  const add = instructions[index - 1];
  const load = instructions[index - 2];
  if (add?.op !== "add" || load?.op !== "movsxd") return [];
  // `add base, entry`: the destination is the entry the load produced, and the source is the table's own address
  const [sum, offset] = add.operands;
  if (sum?.kind !== "reg" || offset?.kind !== "reg") return [];
  const entryRegister = registerNumberOf(sum.name);
  const baseRegister = registerNumberOf(offset.name);
  if (entryRegister === undefined || baseRegister === undefined) return [];
  const [entry, table] = load.operands;
  if (entry?.kind !== "reg" || table?.kind !== "memory") return [];
  if (registerNumberOf(entry.name) !== entryRegister) return [];
  if (table.scale !== 4 || table.index === undefined) return [];
  if (table.base === undefined || registerNumberOf(table.base) !== baseRegister) return [];
  /*
   * The base is usually loaded three instructions before the jump, and gcc loads it much earlier when the function has other work
   * between: the table's address is invariant, so there is no reason to compute it twice. What matters is which instruction last
   * wrote the register, not how far back it sits.
   */
  const lea = definingLea(instructions, index - 2, baseRegister);
  if (lea === undefined) return [];
  const [base, address] = lea.operands;
  if (base?.kind !== "reg" || address?.kind !== "memory" || !address.ripRelative) return [];
  if (registerNumberOf(base.name) !== baseRegister) return [];
  /*
   * How many entries there are comes from the bounds check the compiler put before the table: `cmp $0x22,%edx; ja default` means
   * twenty-three entries, and `and $0xf,%edx` means sixteen. Without one there is no bound to trust, and a table read past its end is
   * a target in some other function, so an unmatched shape names no targets at all.
   */
  const indexRegister = registerNumberOf(table.index);
  let count: number | undefined;
  /*
   * The bound is searched for from the instruction before the jump rather than from before the load: a bounds check sits above the
   * `lea`, but a mask is applied to the extracted index right under the load, three instructions from the jump.
   */
  for (let back = index - 1; back >= 0 && index - back <= 16; back -= 1) {
    const candidate = instructions[back]!;
    if (candidate.op !== "cmp" && candidate.op !== "and") continue;
    const immediate = candidate.operands.find((operand) => operand.kind === "immediate");
    const compared = candidate.operands.find((operand) => operand.kind === "reg");
    if (immediate?.kind !== "immediate" || compared?.kind !== "reg") continue;
    if (registerNumberOf(compared.name) !== indexRegister) continue;
    const bound = Number(immediate.value);
    /*
     * A `cmp` against the highest case is the shape gcc emits for a `switch` over a range, and `bound + 1` entries is what it names.
     * A mask is the other shape: `and $0xf,%eax` keeps the low four bits, so sixteen entries, and it is what a compiler emits rather
     * than a comparison when the value is a field being extracted, as it is in `cpuid` handling. Only a mask of low bits is a count:
     * any other `and` is doing work on the value rather than bounding it.
     */
    if (candidate.op === "and" && ((bound + 1) & bound) !== 0) continue;
    count = bound + 1;
    break;
  }
  if (count === undefined || count < 1 || count > 1024) return [];
  const tableAddress = bodyAddress + BigInt(lea.offset + lea.length) + address.displacement;
  const targets: number[] = [];
  for (let slot = 0n; slot < BigInt(count); slot += 1n) {
    const value = read(tableAddress + slot * 4n, 4);
    if (value === undefined) break;
    const target = tableAddress + BigInt.asIntN(32, value) - bodyAddress;
    if (target < 0n || target >= BigInt(bodyBytes)) break;
    targets.push(Number(target));
  }
  return targets;
}

/**
 * A `jmp` that leaves the body's own bytes, as a call and a return.
 *
 * A compiler emits exactly this for a function whose whole body is a call with the same arguments, and for a stub that computes an
 * argument and hands over to a worker: there is a `jmp` where the source had a call at the end of a function, because reusing the
 * frame is cheaper than making a new one. The two instructions do what the one did: the arguments are already where SysV puts them,
 * the callee's answer lands in `%rax`, and returning it is the return the `jmp` replaced.
 */
function translateTailCall(instruction: Instruction, options: TranslateOptions): number[] {
  const target = options.tailCalls?.get(instruction.offset);
  if (target === undefined) {
    throw new TranslateError(instruction.offset, "a tail call needs the address its jump reaches");
  }
  if (options.calls === undefined) {
    throw new TranslateError(
      instruction.offset,
      "a tail call needs a program to call into: use translateProgram, which follows the target and translates it",
    );
  }
  return [
    OP.call,
    ...unsigned(options.calls.indexOf(target, instruction) + (options.syscalls === true ? 1 : 0)),
    ...globalGet(REGISTERS.indexOf("rax")),
    OP.returnOp,
  ];
}

/** The offset of the last instruction, which with its length is how many bytes the body covers. */
function lastOffset(instructions: readonly Instruction[]): number {
  return instructions.at(-1)?.offset ?? 0;
}

function blockStarts(instructions: readonly Instruction[], bodyAddress: bigint, options: TranslateOptions): number[] {
  const starts = new Set<number>([0]);
  for (const [index, instruction] of instructions.entries()) {
    if (instruction.op === "ret") {
      starts.add(index + 1);
      continue;
    }
    if (!instruction.op.startsWith("j")) continue;
    /*
     * A jump through a value has no target in the bytes, so the only block start it implies is the instruction after it: the bytes
     * from there on can be reached by falling into them, or not at all, but they are not the middle of the block that jumped.
     */
    if (instruction.operands[0]?.kind !== "immediate") {
      starts.add(index + 1);
      const bodyBytes = lastOffset(instructions) + (instructions.at(-1)?.length ?? 0);
      for (const target of jumpTableTargets(instructions, index, bodyAddress, bodyBytes, options)) {
        const targetIndex = instructions.findIndex((entry) => entry.offset === target);
        if (targetIndex >= 0) starts.add(targetIndex);
      }
      continue;
    }
    // a jump out of the body lands in no block of it: what it reaches is a function of its own, called and returned from
    if (options.tailCalls?.has(instruction.offset) === true) {
      starts.add(index + 1);
      continue;
    }
    const target = indexOfOffset(instructions, branchTarget(instruction));
    if (target >= 0) starts.add(target);
    starts.add(index + 1);
  }
  return [...starts].filter((index) => index < instructions.length).sort((lower, higher) => lower - higher);
}

/**
 * Translates a body whose control flow is a graph rather than a tree, by naming its blocks and running whichever block a branch
 * chose.
 *
 * A compiled resolver is the reason this exists: it shares a tail between two paths, jumps back into a block that is also
 * reached by falling into it, and does both inside the same function. WASM cannot jump into the middle of a block, so the graph is
 * flattened: each block is guarded by a comparison against the block number kept in a local, and a branch sets that local and
 * returns to the top. Registers are globals and a comparison's operands are locals, so a block carries no state of its own.
 */
function translateAsBlocks(instructions: readonly Instruction[], options: TranslateOptions): number[] {
  const starts = blockStarts(instructions, bodyBase(options), options);
  const numbers = new Map<number, number>(starts.map((start, number) => [start, number]));
  const blockAt = (index: number): number => {
    const number = numbers.get(index);
    if (number === undefined) {
      throw new TranslateError(
        instructions[Math.min(index, instructions.length - 1)]?.offset ?? 0,
        "a branch lands between blocks, which is not where an instruction starts",
      );
    }
    return number;
  };
  /*
   * The blocks a branch lands in, which are the ones whose flags are whatever the branch that reached them had. A block reached only
   * by falling out of the block before it inherits that block's flags instead: `cmp $0x1c,%edx; je L1; jbe L2` is two branches on one
   * comparison, and the second one is reached by the first one not being taken.
   */
  const targeted = new Set<number>();
  for (const instruction of instructions) {
    if (!instruction.op.startsWith("j") || instruction.operands[0]?.kind !== "immediate") continue;
    targeted.add(indexOfOffset(instructions, branchTarget(instruction)));
  }
  let carried: { from: number; flags: Flags } | undefined;
  const out: number[] = [];
  out.push(...i64Const(0n), ...localSet(PC), OP.loop, OP.emptyBlockType);
  for (const [number, start] of starts.entries()) {
    const end = number + 1 < starts.length ? starts[number + 1]! : instructions.length;
    out.push(...localGet(PC), ...i64Const(BigInt(number)), OP.i64Eq, OP.ifEmpty, OP.emptyBlockType);
    const terminal = end - 1;
    const last = instructions[terminal]!;
    const before = carried;
    const flags: Flags = before !== undefined && before.from === start && !targeted.has(start)
      ? { ...before.flags }
      : { set: false, width: 64 };
    // everything before the block's last instruction is straight-line code, and the empty scope is what says so
    translateRange(instructions, start, terminal, new Map(), out, flags, new Map(), options);
    carried = last.op.startsWith("j") && last.operands[0]?.kind === "immediate" ? { from: end, flags: { ...flags } } : undefined;
    if (last.op === "ret") {
      out.push(...globalGet(REGISTERS.indexOf("rax")), OP.returnOp);
    } else if (last.op === "jmp" && last.operands[0]?.kind !== "immediate") {
      /*
       * A jump whose target is a value: the blocks this body knows are the blocks it can land in, so the target is compared against
       * each block's address and the one that matches becomes the block to run. A target that matches nothing is `unreachable`,
       * which is the trap a jump to an address that is not an instruction is.
       *
       * This is not how a `switch` is translated yet: its cases are reached only through a table in the image, which the walk does
       * not read, so the case bodies are not blocks here and a jump into one traps.
       */
      const operand = last.operands[0];
      out.push(...(operand?.kind === "memory" ? loadFromMemory(operand, 64, last) : read(operand!, last)), ...localSet(INDIRECT));
      out.push(OP.block, OP.emptyBlockType);
      for (const [number, start] of starts.entries()) {
        const address = bodyBase(options) + BigInt(instructions[start]!.offset);
        out.push(
          ...localGet(INDIRECT), ...i64Const(address), OP.i64Eq, OP.ifEmpty, OP.emptyBlockType,
          ...i64Const(BigInt(number)), ...localSet(PC),
          OP.br, ...unsigned(1),
          OP.end,
        );
      }
      out.push(OP.unreachable, OP.end);
    } else if (last.op === "jmp" && options.tailCalls?.has(last.offset) === true) {
      out.push(...translateTailCall(last, options));
    } else if (last.op === "jmp") {
      out.push(...i64Const(BigInt(blockAt(indexOfOffset(instructions, branchTarget(last))))), ...localSet(PC));
    } else if (last.op.startsWith("j")) {
      const suffix = last.op.slice(1);
      if (suffix === "o" || suffix === "no" || suffix === "p" || suffix === "np") {
        throw new TranslateError(last.offset, `j${suffix} reads the overflow or parity flag, which this does not compute`);
      }
      if (!flags.set) {
        throw new TranslateError(last.offset, `${last.op} needs a cmp or test that every path reaching it goes through`);
      }
      const taken = blockAt(indexOfOffset(instructions, branchTarget(last)));
      if (end >= instructions.length) {
        throw new TranslateError(last.offset, `${last.op} falls off the end of the function, which has no value to return`);
      }
      out.push(
        ...flagCondition(suffix, flags, last),
        OP.ifEmpty,
        OP.emptyBlockType,
        ...i64Const(BigInt(taken)),
        ...localSet(PC),
        OP.elseOp,
        ...i64Const(BigInt(blockAt(end))),
        ...localSet(PC),
        OP.end,
      );
    } else {
      // the block ends because a branch somewhere else lands on the next instruction, so this one runs and the next is chosen
      translateRange(instructions, terminal, end, new Map(), out, flags, new Map(), options);
      if (end >= instructions.length) {
        throw new TranslateError(last.offset, "the last block runs off the end of the function, which has no value to return");
      }
      out.push(...i64Const(BigInt(blockAt(end))), ...localSet(PC));
    }
    out.push(OP.end);
  }
  // the dispatch loop is only left by a `return`, so what follows it is unreachable and says so
  out.push(OP.br, ...unsigned(0), OP.end, OP.unreachable);
  return out;
}

/**
 * Translates one function's instructions into a WASM body.
 *
 * Only the entry function sets `%rsp`: it is the machine's initial state, and a callee that reset it would throw away the
 * stack its caller was using. Every function returns `%rax`, which is the whole of this calling convention.
 */
function bodyOf(
  instructions: readonly Instruction[],
  options: TranslateOptions,
): number[] {
  if (instructions.length === 0) {
    throw new TranslateError(0, "there is nothing to translate");
  }
  const resolved = resolveAddresses(instructions, options);
  const body: number[] = [];
  const flags: Flags = { set: false, width: 64 };
  const loops = findLoops(resolved);
  /*
   * The stack's pointer is set here rather than by the caller, because `%rsp` is a register and registers start at zero.
   * Pointing it at the top of the imported memory is the one decision that makes every later push and pop arithmetic rather
   * than a special case, and it happens before the body so a stack access in the first instruction is already legal.
   */
  if (usesStack(resolved)) {
    if ((options.stackPages ?? 0) < 1) {
      throw new TranslateError(
        0,
        "this function uses the stack, so it needs stackPages: pass how many pages to set aside above the image",
      );
    }

  }
  if (needsDispatcher(resolved, options)) {
    // a graph rather than a tree: blocks and a block number, not nesting. That body ends in `unreachable`, so it is complete.
    body.push(...translateAsBlocks(resolved, options));
    return body;
  }
  translateRange(resolved, 0, resolved.length, new Map(), body, flags, loops, options);
  /*
   * The block has to end with a value, because every function here returns `i64`. Three cases leave nothing on the stack: a
   * `ret`, which returned already; a `jmp`, after which the fall-through is dead code and the validator is satisfied with the
   * branch; and the two instructions that end a program rather than a block. `syscall` leaves `%rax` holding the call's answer,
   * and a trailing `call` does too, whether or not it comes back.
   */
  if (resolved.at(-1)?.op !== "ret" && resolved.at(-1)?.op !== "jmp") {
    /*
     * A program's entry point usually ends in `exit`, which never comes back, so there is no value to return with. What it
     * leaves is `%rax`, where the call's answer would have gone. Ending in anything else is a truncated function, and that is
     * refused: guessing a value there would hide the truncation behind a plausible number.
     */
    // there is nothing else a live fall-through can end with, so anything else is the truncation this refuses
    /*
     * A trailing `call` is the other end a block can have: the instruction before the padding is a call to a function that never
     * returns, so nothing follows it in this function's bytes. Returning `%rax` is right there too, because that is where a
     * callee's answer would have gone if it came back.
     */
    /*
     * A `hlt` is the other way a body ends without returning: it traps, so nothing follows it, and what it leaves in `%rax` is where a
     * `call`'s answer would have gone. An entry point that calls `exit` and then halts is exactly that shape.
     */
    const last = resolved.at(-1)?.op;
    if (last !== "syscall" && last !== "call" && last !== "hlt") {
      throw new TranslateError(
        resolved.at(-1)?.offset ?? 0,
        `the body at ${hex(bodyBase(options))} does not end in ret, in a syscall, in a hlt or in a call that never ` +
          "returns, so it has no value to return",
      );
    }
    body.push(...globalGet(REGISTERS.indexOf("rax")));
  }
  return body;
}

/**
 * Translates a program: the entry function, and every function it calls, each as its own WASM function.
 *
 * This is the increment that turns the translator from a calculator into a machine. The registers are module globals, so a
 * call passes its arguments exactly as SysV says: by writing `%rdi`, `%rsi`, `%rdx` and the rest. `call` is a WASM `call`,
 * `ret` is a WASM `return`, and the result comes back in `%rax`. A callee is found by following the call's displacement into
 * the code region the caller supplied, which is why that region is required the moment anything is called.
 */
function programOf(
  entryInstructions: readonly Instruction[],
  name: string,
  options: TranslateOptions,
): Uint8Array {
  const entryAddress = options.codeAddress ?? options.code?.virtualAddress ?? 0n;
  // `order` is both the queue and the module index: position i is function i, and a call appends on first sight
  const order: Body[] = [];
  const known = new Map<string, number>();

  const register = (
    address: bigint,
    instructions: readonly Instruction[],
    tailCalls: ReadonlyMap<number, bigint> = new Map(),
    traps = false,
  ): number => {
    const key = address.toString();
    const seen = known.get(key);
    if (seen !== undefined) return seen;
    const index = order.length;
    known.set(key, index);
    order.push({ address, instructions, tailCalls, traps });
    return index;
  };

  /**
   * Whether a processor would execute the byte at this address. Without `executable` every byte of the region counts, which is the
   * standing answer for a caller that gave bytes and no segments.
   */
  const isExecutable = (address: bigint): boolean =>
    options.executable === undefined ||
    options.executable.some((range) => range.start <= address && address < range.end);

  const calls: CallRegistry = {
    candidates: () => order.map((body) => body.address),
    indexOf(address, instruction) {
      const seen = known.get(address.toString());
      if (seen !== undefined) return seen;
      const region = options.code;
      if (region === undefined) {
        throw new TranslateError(
          instruction.offset,
          `call to ${hex(address)} needs the code region it was cut from: pass code: { bytes, virtualAddress }`,
        );
      }
      /*
       * A target no segment would execute is not a function, and its bytes are not instructions: a header, a relocation table or
       * data. Decoding them is how a static PIE's `call 0` turned the ELF header into a body, so the body registered for such an
       * address is a trap. A call to it keeps the program's own answer, since the path that makes it is the one that crashes.
       */
      if (!isExecutable(address)) return register(address, [], new Map(), true);
      const offset = address - region.virtualAddress;
      if (offset < 0n || offset >= BigInt(region.bytes.length)) {
        throw new TranslateError(
          instruction.offset,
          `call to ${hex(address)} is outside the code region, so there is nothing here to translate: an external or ` +
            `library function needs the syscall layer, which does not exist yet`,
        );
      }
      const body = region.bytes.subarray(Number(offset));
      try {
        const walked = functionExtent(body, address, options);
        return register(address, walked.instructions, walked.tailCalls);
      } catch (error) {
        /*
         * The walk's own refusals name the function they are about, but one thrown while decoding a byte does not: the offset it
         * carries is relative to a body the caller never asked for by name. Naming the function here is what turns "one of the
         * functions this program calls cannot be decoded" into an address to look at.
         */
        if (error instanceof DecodeError) {
          throw new TranslateError(error.offset, `the function at ${hex(address)}: ${error.detail}`);
        }
        throw error;
      }
    },
  };

  // index 0 is the entry, registered before anything can call it, so a recursive call resolves to the function itself
  register(entryAddress, entryInstructions);
  const withCalls: TranslateOptions = { ...options, calls };
  const bodies: number[][] = [];
  /*
   * Twice: an indirect call is a comparison against every function this program translated, so the list has to be complete before
   * any body is emitted, and translating the bodies is what completes it. The first pass grows the list and is thrown away; the
   * second emits the same bodies against a list that cannot grow any further, because the same bytes make the same decisions.
   */
  for (let pass = 0; pass < 2; pass += 1) {
    bodies.length = 0;
    for (let index = 0; index < order.length; index += 1) {
      const body = order[index]!;
      // a body that traps needs no translation: `unreachable` satisfies any result type, and it is what jumping there does
      bodies.push(
        body.traps === true
          ? [OP.unreachable]
          : bodyOf(body.instructions, { ...withCalls, functionAddress: body.address, tailCalls: body.tailCalls }),
      );
    }
  }

  const all = order.flatMap((body) => [...body.instructions]);
  const stackUsed =
    all.some((instruction) => instruction.op === "push" || instruction.op === "pop" || instruction.op === "leave") ||
    all.some((instruction) =>
      instruction.operands.some(
        (operand) =>
          operand.kind === "memory" && [operand.base, operand.index].some((register) => register === "rsp" || register === "rbp"),
      ),
    );
  const touchesMemory =
    stackUsed || all.some((instruction) => instruction.operands.some((operand) => operand.kind === "memory"));
  /*
   * The stack's pointer is machine state, so it is set once, in the entry body, and only when *anything* in the program uses
   * the stack: a callee that pushes is enough, since its caller's frame is the same stack.
   */
  if (stackUsed) {
    const { top } = stackBounds(options);
    bodies[0]!.unshift(...i64Const(top), ...globalSet(REGISTERS.indexOf("rsp")));
  }
  return module({
    name,
    locals: LOCAL_COUNT,
    globals: REGISTERS.length,
    // the thread pointer sits after the register file and is exported, so the host can point %fs somewhere else at run time
    threadPointer: { name: "fs" },
    ...(options.syscalls === true ? { importFunction: { module: "env", field: "syscall" } } : {}),
    bodies,
    // the import appears only when something addresses memory, so a module that reads nothing still needs nothing
    ...(touchesMemory ? { memory: { minimumPages: totalPages(options) } } : {}),
  });
}

/** Bytes to a module, which is the path every test uses: decode, translate, run. */
export function translateProgram(
  bytes: Uint8Array,
  name = "main",
  options: TranslateOptions = {},
): Uint8Array {
  const instructions = decode(bytes);
  refuseCallInsideBody(instructions, bytes.length, options.codeAddress ?? 0n);
  return named(options, instructions, () => programOf(instructions, name, options));
}

/**
 * Refuses a call whose target is inside the body being translated but is not one of its instruction starts.
 *
 * The decoder will happily decode from the middle of an instruction, and the bits it produces are a program that loads, runs
 * and computes something else. So the question is not whether the target decodes, it is whether it is where an instruction
 * begins, which is a property of this body and not of the bytes.
 */
function refuseCallInsideBody(
  instructions: readonly Instruction[],
  length: number,
  address: bigint,
): void {
  const starts = new Set(instructions.map((instruction) => instruction.offset));
  for (const instruction of instructions) {
    if (instruction.op !== "call") continue;
    const target = branchTarget(instruction);
    if (target >= 0 && target < length && !starts.has(target)) {
      throw new TranslateError(
        instruction.offset,
        `the call at ${hex(address + BigInt(instruction.offset))} targets ${hex(address + BigInt(target))}, which is inside ` +
          "this body and is not an instruction start, so it is not where a function starts",
      );
    }
  }
}

/** The same thing under the name the earlier tests use, which is also the name the README uses. */
export function translateBytes(
  bytes: Uint8Array,
  name = "main",
  options: TranslateOptions = {},
): Uint8Array {
  return programOf(decode(bytes), name, options);
}

/**
 * Translates one function, with nothing to call into.
 *
 * Kept for the common case and for the tests: a `call` is refused here because there is no program for it to call into, and
 * `translateProgram` is the function that follows one.
 */
export function translate(
  instructions: readonly Instruction[],
  name = "main",
  options: TranslateOptions = {},
): Uint8Array {
  // the same refusal a decoded byte slice gets, because a list of instructions can be walked into the middle of one too
  const length = instructions.reduce((furthest, instruction) => Math.max(furthest, instruction.offset + instruction.length), 0);
  refuseCallInsideBody(instructions, length, options.codeAddress ?? 0n);
  return named(options, instructions, () => programOf(instructions, name, options));
}

/**
 * Names the function a refusal happened in, which the offsets alone cannot: an offset is measured from the body's first byte, and a
 * program translated body by body has dozens of them. A message that already names the function is left as it is.
 */
function named(
  options: TranslateOptions,
  instructions: readonly Instruction[],
  translateBody: () => Uint8Array,
): Uint8Array {
  try {
    return translateBody();
  } catch (error) {
    // the body being translated is the function's own address when it has one, which is where its offsets are measured from
    const address = bodyBase(options);
    if (!(error instanceof TranslateError) || address === undefined || error.message.includes("the function at")) throw error;
    // an offset that is not one of this body's belongs to a callee, whose own translation names it
    if (!instructions.some((instruction) => instruction.offset === error.offset)) throw error;
    throw new TranslateError(error.offset, `in the function at ${hex(address)}: ${error.message}`);
  }
}
