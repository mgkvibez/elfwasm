/**
 * The translator: decoded instructions in, a WASM module out.
 *
 * Stage 3 starts where the wrong abstraction would hurt: one straight-line block, no branches, no flags, no memory. That
 * is not a toy subset picked for convenience; it is the smallest thing whose correctness can be checked end to end, by
 * running the module and comparing the number it returns against the number the same bytes compute natively. Every
 * instruction outside it is refused by name and offset, in the same spirit as the decoder, because a translator that
 * silently drops an instruction returns a wrong answer with no trace of where it went wrong.
 *
 * Register layout: the sixteen 64-bit registers are sixteen i64 locals in encoding order, `rax` to `r15`. A 32-bit
 * operand names one of those locals and carries width 32, which is x86's own model: writing the low 32 bits of a register
 * zeroes the upper 32, and reading it ignores them. The translation makes that explicit rather than leaving it to
 * coincidence, because an i64 local that keeps its high bits through a 32-bit write is the classic way to turn an
 * `int` result into a different `long`.
 */

import { decode, type Instruction, type Operand } from "./decode.js";
import { I64, module, i64Const, i32Const, localGet, localSet, OP } from "./emit.js";

/** The register file, in the order the encoding numbers it. Index is the WASM local. */
const REGISTERS = [
  "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
] as const;

export class TranslateError extends Error {
  readonly offset: number;

  constructor(offset: number, message: string) {
    super(`at offset 0x${offset.toString(16)}: ${message}`);
    this.offset = offset;
  }
}

function localOf(operand: Operand, instruction: Instruction): number {
  if (operand.kind !== "reg") {
    throw new TranslateError(instruction.offset, `expected a register operand, found ${operand.kind}`);
  }
  const base = operand.name.replace(/^r(\d+)[bwd]?$/, (_match, digits: string) => `r${digits}`);
  const index = REGISTERS.indexOf(base as (typeof REGISTERS)[number]);
  if (index < 0) {
    throw new TranslateError(instruction.offset, `register ${operand.name} is not one of the 64-bit registers`);
  }
  return index;
}

/** A register read, as the i64 the local holds. Width is applied by the caller, which knows the operation's width. */
function read(operand: Operand, instruction: Instruction): number[] {
  return localGet(localOf(operand, instruction));
}

function write(operand: Operand, instruction: Instruction): number[] {
  return localSet(localOf(operand, instruction));
}

/**
 * Narrows a value to 32 bits and widens it back, which is what a 32-bit operation does to its result.
 *
 * The widening is unsigned on purpose: a 32-bit write zeroes the upper half in x86-64, so `mov $-1, %eax` leaves `rax`
 * holding `0x00000000ffffffff` and not `-1`. Using the signed form here would be a one-instruction bug that shows up only
 * when a value is negative.
 */
const NARROW_32 = OP.i32WrapI64;
const WIDEN_32 = OP.i64ExtendI32U;

/** The i32 opcodes for the arithmetic, so a 32-bit operation is computed at its own width rather than at 64. */
const BINARY = {
  add: { bits64: OP.i64Add, bits32: 0x6a },
  sub: { bits64: OP.i64Sub, bits32: 0x6b },
  xor: { bits64: OP.i64Xor, bits32: 0x73 },
  and: { bits64: OP.i64And, bits32: 0x71 },
  or: { bits64: OP.i64Or, bits32: 0x72 },
  imul: { bits64: OP.i64Mul, bits32: 0x6c },
} as const;

type BinaryOp = keyof typeof BINARY;

function translateMov(instruction: Instruction): number[] {
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, "mov needs two operands");
  }
  if (destination.kind !== "reg") {
    // a store into memory is the next increment; refusing it is what keeps this one honest
    throw new TranslateError(instruction.offset, "a mov into memory is not translated yet");
  }
  const width = destination.bits;
  if (source.kind === "immediate") {
    // a 32-bit destination zero-extends the immediate into the 64-bit register, which is why this masks rather than
    // sign-extends: `mov $-1, %eax` leaves rax holding 0xffffffff
    const value = width === 32 ? BigInt.asUintN(32, source.value) : source.value;
    if (width === 8 || width === 16) {
      throw new TranslateError(instruction.offset, `a ${width}-bit mov is not translated yet`);
    }
    return [...i64Const(value), ...write(destination, instruction)];
  }
  if (source.kind === "memory") {
    throw new TranslateError(instruction.offset, "a mov from memory is not translated yet");
  }
  const code = read(source, instruction);
  // a 32-bit write zeroes the upper half, so the value is narrowed and widened rather than copied
  if (width === 32) return [...code, NARROW_32, WIDEN_32, ...write(destination, instruction)];
  return [...code, ...write(destination, instruction)];
}

function translateBinary(instruction: Instruction): number[] {
  const op = instruction.op as BinaryOp;
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
  }
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} into memory is not translated yet`);
  }
  const width = destination.bits;
  if (width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} is not translated yet`);
  }

  let operand: number[];
  if (source.kind === "immediate") {
    operand = width === 32 ? i32Const(Number(BigInt.asIntN(32, source.value))) : i64Const(source.value);
  } else if (source.kind === "memory") {
    throw new TranslateError(instruction.offset, `a ${instruction.op} from memory is not translated yet`);
  } else {
    operand = read(source, instruction);
  }

  const opcodes = BINARY[op];
  if (width === 64) {
    return [...read(destination, instruction), ...operand, opcodes.bits64, ...write(destination, instruction)];
  }
  // at 32 bits the operation happens in i32, then widens back unsigned: the add is 32-bit, and its result's upper half is
  // zero rather than a copy of bit 31
  return [
    ...read(destination, instruction),
    NARROW_32,
    ...operand,
    opcodes.bits32,
    WIDEN_32,
    ...write(destination, instruction),
  ];
}

const TRANSLATED = new Set(["mov", "add", "sub", "xor", "and", "or", "imul"]);

/**
 * Translates one straight-line block.
 *
 * The block must end in `ret`, and the value it returns is `rax`, which is the only calling convention this increment
 * claims. Anything else, including a branch, is refused: a translator that guessed at control flow would produce a module
 * that runs and computes the wrong thing, which is worse than one that refuses.
 */
export function translate(instructions: readonly Instruction[], name = "main"): Uint8Array {
  const body: number[] = [];
  let returned = false;

  for (const instruction of instructions) {
    if (returned) {
      throw new TranslateError(instruction.offset, "there is code after ret, which needs a block this cannot translate");
    }
    if (instruction.op === "nop" || instruction.op === "endbr64") continue;
    if (instruction.op === "ret") {
      if (instruction.operands.some((operand) => operand.kind === "immediate")) {
        throw new TranslateError(instruction.offset, "ret with a stack adjustment is not translated yet");
      }
      // the function's result is rax, which the module returns as its i64
      body.push(...localGet(REGISTERS.indexOf("rax")));
      returned = true;
      continue;
    }
    if (!TRANSLATED.has(instruction.op)) {
      throw new TranslateError(instruction.offset, `${instruction.op} is not translated yet`);
    }
    body.push(...(instruction.op === "mov" ? translateMov(instruction) : translateBinary(instruction)));
  }

  if (!returned) {
    throw new TranslateError(
      instructions.at(-1)?.offset ?? 0,
      "the block does not end in ret, so it has no value to return",
    );
  }
  return module({ name, locals: REGISTERS.length, body, result: I64 });
}

/** Bytes to a module, which is the path every test uses: decode, translate, run. */
export function translateBytes(bytes: Uint8Array, name = "main"): Uint8Array {
  return translate(decode(bytes), name);
}
