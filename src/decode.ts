/**
 * The x86-64 decoder: bytes in, a flat instruction list out.
 *
 * Stage 2 of the roadmap, and it comes before the translator because a translator written against a guess about what a
 * compiler emits is a translator that works on the guess and nothing else. The subset here is what a statically linked
 * musl `hello world` uses in its own code and in the libc functions it calls: moves, `lea`, the arithmetic that sets
 * flags, `cmp`, calls, conditional jumps, `syscall` and `ret`. Anything else is refused with its byte offset, because a
 * decoder that skips an unknown instruction hands the translator a code stream that is subtly out of phase, and the
 * resulting program is wrong in a way no test here would catch.
 *
 * The IR is deliberately flat: a mnemonic, its operands, and where it started. No flags object, no implicit operands, no
 * instruction fusion. Those are translator decisions, and the translator is Stage 3.
 */

export type Register =
  | "rax"
  | "rcx"
  | "rdx"
  | "rbx"
  | "rsp"
  | "rbp"
  | "rsi"
  | "rdi"
  | "r8"
  | "r9"
  | "r10"
  | "r11"
  | "r12"
  | "r13"
  | "r14"
  | "r15";

export type Operand =
  | { readonly kind: "reg"; readonly name: Register; readonly bits: 32 | 64 }
  | { readonly kind: "immediate"; readonly value: bigint }
  | {
      readonly kind: "memory";
      readonly base?: Register;
      readonly index?: Register;
      readonly scale: 1 | 2 | 4 | 8;
      /** Signed displacement, or the signed target of a RIP-relative reference. */
      readonly displacement: bigint;
      /** True for `[rip + disp32]`, whose address depends on the instruction's own length. */
      readonly ripRelative: boolean;
    };

export interface Instruction {
  readonly offset: number;
  /** Encoded length, so a caller can check the decoder consumed exactly the bytes it claims. */
  readonly length: number;
  /** `mov`, `lea`, `xor`, `add`, `sub`, `cmp`, `push`, `pop`, `call`, `jmp`, `je`, `jne`, `ret`, `syscall`, `nop`. */
  readonly op: string;
  readonly operands: readonly Operand[];
}

export class DecodeError extends Error {
  readonly offset: number;

  constructor(offset: number, message: string) {
    super(`at offset 0x${offset.toString(16)}: ${message}`);
    this.name = "DecodeError";
    this.offset = offset;
  }
}

const REGISTERS: readonly Register[] = [
  "rax",
  "rcx",
  "rdx",
  "rbx",
  "rsp",
  "rbp",
  "rsi",
  "rdi",
  "r8",
  "r9",
  "r10",
  "r11",
  "r12",
  "r13",
  "r14",
  "r15",
];

interface Cursor {
  readonly bytes: Uint8Array;
  readonly start: number;
  index: number;
}

interface Rex {
  readonly w: boolean;
  readonly r: boolean;
  readonly x: boolean;
  readonly b: boolean;
}

function byteAt(cursor: Cursor): number {
  if (cursor.index >= cursor.bytes.length) {
    throw new DecodeError(cursor.start, "the instruction runs past the end of the code buffer");
  }
  return cursor.bytes[cursor.index++]!;
}

function readImmediate(cursor: Cursor, size: 1 | 2 | 4 | 8): bigint {
  let value = 0n;
  for (let index = 0; index < size; index += 1) value |= BigInt(byteAt(cursor)) << BigInt(8 * index);
  const bits = BigInt(size * 8);
  const signBit = 1n << (bits - 1n);
  return (value & signBit) === 0n ? value : value - (1n << bits);
}

function registerOperand(code: number, extended: boolean, bits: 32 | 64): Operand {
  return { kind: "reg", name: REGISTERS[code | (extended ? 8 : 0)]!, bits };
}

interface ModRm {
  readonly mode: number;
  readonly regField: number;
  readonly rm: Operand;
}

function decodeModRm(cursor: Cursor, rex: Rex, bits: 32 | 64): ModRm {
  const byte = byteAt(cursor);
  const mode = byte >> 6;
  const regField = (byte >> 3) & 0x7;
  const rmField = byte & 0x7;

  if (mode === 3) return { mode, regField, rm: registerOperand(rmField, rex.b, bits) };

  let base: Register | undefined;
  let index: Register | undefined;
  let scale: 1 | 2 | 4 | 8 = 1;
  let displacement = 0n;
  let ripRelative = false;

  if (rmField === 4) {
    const sib = byteAt(cursor);
    scale = (1 << ((sib >> 6) & 0x3)) as 1 | 2 | 4 | 8;
    const indexField = (sib >> 3) & 0x7;
    const baseField = sib & 0x7;
    // index 4 without REX.X means "no index", not rsp, which cannot be an index register at all
    if (!(indexField === 4 && !rex.x)) index = REGISTERS[indexField | (rex.x ? 8 : 0)]!;
    if (baseField === 5 && mode === 0) {
      displacement = readImmediate(cursor, 4);
    } else {
      base = REGISTERS[baseField | (rex.b ? 8 : 0)]!;
      if (mode === 1) displacement = readImmediate(cursor, 1);
      else if (mode === 2) displacement = readImmediate(cursor, 4);
    }
  } else if (rmField === 5 && mode === 0) {
    // mod 00, rm 101 is RIP-relative in 64-bit mode, the one addressing form with no base register
    displacement = readImmediate(cursor, 4);
    ripRelative = true;
  } else {
    base = REGISTERS[rmField | (rex.b ? 8 : 0)]!;
    if (mode === 1) displacement = readImmediate(cursor, 1);
    else if (mode === 2) displacement = readImmediate(cursor, 4);
  }

  const memory: Operand = {
    kind: "memory",
    ...(base === undefined ? {} : { base }),
    ...(index === undefined ? {} : { index }),
    scale,
    displacement,
    ripRelative,
  };
  return { mode, regField, rm: memory };
}

// group 1 by /reg field: 0 add, 1 or, 2 adc, 3 sbb, 4 and, 5 sub, 6 xor, 7 cmp. The carry forms are left out on
// purpose: nothing a compiler emits for C code needs them, and a translator that ignored the carry would be wrong.
const ARITHMETIC: readonly (string | undefined)[] = ["add", "or", undefined, undefined, "and", "sub", "xor", "cmp"];

function decodeOne(bytes: Uint8Array, offset: number): Instruction {
  const cursor: Cursor = { bytes, start: offset, index: offset };
  let rex: Rex = { w: false, r: false, x: false, b: false };

  for (;;) {
    const next = bytes[cursor.index];
    if (next === undefined) throw new DecodeError(offset, "the code buffer ends inside an instruction");
    if (next === 0x66) {
      throw new DecodeError(offset, "a 16-bit operand size prefix (0x66) is not in the decoded subset");
    }
    if (next === 0x2e || next === 0x36 || next === 0x3e || next === 0x26 || next === 0x64 || next === 0x65) {
      throw new DecodeError(offset, "a segment override prefix is not in the decoded subset");
    }
    if (next === 0xf0 || next === 0xf2 || next === 0xf3) {
      throw new DecodeError(offset, "a lock or repeat prefix is not in the decoded subset");
    }
    if (next >= 0x40 && next <= 0x4f) {
      cursor.index += 1;
      rex = { w: (next & 0x8) !== 0, r: (next & 0x4) !== 0, x: (next & 0x2) !== 0, b: (next & 0x1) !== 0 };
      continue;
    }
    break;
  }

  const bits: 32 | 64 = rex.w ? 64 : 32;
  const opcode = byteAt(cursor);
  const finish = (op: string, operands: readonly Operand[]): Instruction => ({
    offset,
    length: cursor.index - offset,
    op,
    operands,
  });

  if (opcode === 0x90) return finish("nop", []);
  if (opcode === 0xc3) return finish("ret", []);
  if (opcode === 0x0f) {
    const second = byteAt(cursor);
    if (second === 0x05) return finish("syscall", []);
    if (second === 0x84) return finish("je", [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
    if (second === 0x85) return finish("jne", [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
    throw new DecodeError(offset, `two-byte opcode 0x0f 0x${second.toString(16)} is not in the decoded subset`);
  }
  if (opcode >= 0x50 && opcode <= 0x57) return finish("push", [registerOperand(opcode - 0x50, rex.b, 64)]);
  if (opcode >= 0x58 && opcode <= 0x5f) return finish("pop", [registerOperand(opcode - 0x58, rex.b, 64)]);
  if (opcode >= 0xb8 && opcode <= 0xbf) {
    // 0x48 B8+rd carries a full 64-bit immediate; without REX.W the immediate is 32 bits and zero-extends
    return finish("mov", [
      registerOperand(opcode - 0xb8, rex.b, bits),
      { kind: "immediate", value: readImmediate(cursor, rex.w ? 8 : 4) },
    ]);
  }
  if (opcode === 0xeb) return finish("jmp", [{ kind: "immediate", value: readImmediate(cursor, 1) }]);
  if (opcode === 0x74) return finish("je", [{ kind: "immediate", value: readImmediate(cursor, 1) }]);
  if (opcode === 0x75) return finish("jne", [{ kind: "immediate", value: readImmediate(cursor, 1) }]);
  if (opcode === 0xe8) return finish("call", [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
  if (opcode === 0xe9) return finish("jmp", [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
  if (opcode === 0x8d) {
    const modrm = decodeModRm(cursor, rex, bits);
    if (modrm.rm.kind !== "memory") {
      throw new DecodeError(
        offset,
        "lea with a register operand cannot be encoded, so these bytes are not what they claim",
      );
    }
    return finish("lea", [registerOperand(modrm.regField, rex.r, bits), modrm.rm]);
  }
  if (opcode === 0x89 || opcode === 0x8b) {
    const modrm = decodeModRm(cursor, rex, bits);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x89 ? finish("mov", [modrm.rm, register]) : finish("mov", [register, modrm.rm]);
  }
  if (opcode === 0x31 || opcode === 0x33) {
    const modrm = decodeModRm(cursor, rex, bits);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x31 ? finish("xor", [modrm.rm, register]) : finish("xor", [register, modrm.rm]);
  }
  if (opcode === 0x01 || opcode === 0x29) {
    const modrm = decodeModRm(cursor, rex, bits);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x01 ? finish("add", [modrm.rm, register]) : finish("sub", [modrm.rm, register]);
  }
  if (opcode === 0xc7) {
    const modrm = decodeModRm(cursor, rex, bits);
    if (modrm.regField !== 0) {
      throw new DecodeError(offset, "0xc7 with a /reg field other than 0 is not a mov and is not in the decoded subset");
    }
    return finish("mov", [modrm.rm, { kind: "immediate", value: readImmediate(cursor, 4) }]);
  }
  if (opcode === 0x83 || opcode === 0x81) {
    const modrm = decodeModRm(cursor, rex, bits);
    const mnemonic = ARITHMETIC[modrm.regField];
    if (mnemonic === undefined) {
      throw new DecodeError(
        offset,
        `arithmetic /${modrm.regField} (0x${opcode.toString(16)}) is not in the decoded subset`,
      );
    }
    return finish(mnemonic, [modrm.rm, { kind: "immediate", value: readImmediate(cursor, opcode === 0x83 ? 1 : 4) }]);
  }

  throw new DecodeError(offset, `opcode 0x${opcode.toString(16)} is not in the decoded subset`);
}

/**
 * Decodes a code buffer from its start to its end.
 *
 * Every byte is accounted for. An instruction the subset does not cover is an error at its offset rather than a skipped
 * byte, because a caller that resumes past an unknown instruction is guessing where the next one begins.
 */
export function decode(bytes: Uint8Array): Instruction[] {
  const instructions: Instruction[] = [];
  let offset = 0;
  while (offset < bytes.length) {
    const instruction = decodeOne(bytes, offset);
    instructions.push(instruction);
    offset += instruction.length;
  }
  return instructions;
}

/** Renders one instruction the way a listing does, so a decode can be read by eye in a test failure. */
export function formatInstruction(instruction: Instruction): string {
  const render = (operand: Operand): string => {
    if (operand.kind === "reg") return operand.name;
    if (operand.kind === "immediate") return operand.value.toString();
    const parts: string[] = [];
    if (operand.ripRelative) parts.push("rip");
    if (operand.base !== undefined) parts.push(operand.base);
    if (operand.index !== undefined) parts.push(`${operand.scale === 1 ? "" : `${operand.scale}*`}${operand.index}`);
    if (operand.displacement === 0n) return parts.length === 0 ? "[0]" : `[${parts.join("+")}]`;
    const magnitude = operand.displacement < 0n ? -operand.displacement : operand.displacement;
    const sign = operand.displacement < 0n ? "-" : "+";
    return `[${parts.join("+")}${sign}0x${magnitude.toString(16)}]`;
  };
  return [instruction.op, ...instruction.operands.map(render)].join(" ");
}
