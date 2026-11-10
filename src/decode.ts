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

/**
 * Byte and word register names, which the wider operands cannot spell.
 *
 * A byte register's name depends on whether REX is present, not only on its code: without REX, codes 4 to 7 are `ah`,
 * `ch`, `dh` and `bh`, and with it they are `spl`, `bpl`, `sil` and `dil`. Reading them from the 64-bit table would name
 * the wrong register and, worse, do it silently. Word registers need a prefix this decoder refuses, so the table exists
 * for completeness and the only way to reach it today is a memory operand.
 */
const BYTE_REGISTERS = [
  "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil",
  "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
] as const;
const HIGH_BYTE_REGISTERS = ["ah", "ch", "dh", "bh"] as const;
const WORD_REGISTERS = [
  "ax", "cx", "dx", "bx", "sp", "bp", "si", "di",
  "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
] as const;

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

export type RegisterName =
  | Register
  | (typeof BYTE_REGISTERS)[number]
  | (typeof HIGH_BYTE_REGISTERS)[number]
  | (typeof WORD_REGISTERS)[number];

export type Operand =
  | { readonly kind: "reg"; readonly name: RegisterName; readonly bits: 8 | 16 | 32 | 64 }
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
      /**
       * The segment prefix, when there is one.
       *
       * `fs` is the one that matters in practice: `mov %fs:0x28, %rax` is the stack canary's read, and a translator that
       * dropped the prefix would turn a thread-local read into a read of address 0x28, which is a plausible looking load
       * of the wrong memory. Keeping it in the IR is what makes that refusal-or-support decision explicit.
       */
      readonly segment?: "fs" | "gs" | "cs" | "ss" | "ds" | "es";
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
  /**
   * Whether a REX prefix was present at all.
   *
   * Not derivable from the four bits: `0x40` is REX with nothing set, and it is exactly that prefix which turns `ah` into
   * `spl`. Reading byte registers without it names the wrong register.
   */
  readonly present: boolean;
}

/**
 * Whether a `0f` map opcode is one of the SSE ones.
 *
 * The ranges are the ones the SIMD encodings occupy, and the point of naming them is the error message: "not in the
 * decoded subset" and "this is SSE" send a reader to different places, and only one of them is true here. The integer
 * opcodes the subset handles (`0f 05`, `0x0b`, `0x1e`, `0x1f`, `0x80` to `0x9f`, `0xaf`, `0xb6`, `0xb7`, `0xbe`, `0xbf`)
 * are checked before this predicate is reached.
 */
function isSseOpcode(second: number): boolean {
  return (
    (second >= 0x10 && second <= 0x17) ||
    (second >= 0x28 && second <= 0x2f) ||
    second === 0x6e ||
    second === 0x7e ||
    (second >= 0x50 && second <= 0x7f) ||
    (second >= 0xc2 && second <= 0xc6) ||
    second >= 0xd0
  );
}

/**
 * Reads the prefix run of an alignment NOP, returning the index of its 0f 1f, or undefined if this is not one.
 *
 * The run must contain at least one prefix and then `0f 1f`; anything else is left to the ordinary prefix handling, which
 * refuses it by name. Nothing is consumed here, so a refusal still reports the offset the prefix started at.
 */
function alignmentNopPrefixes(bytes: Uint8Array, start: number): number | undefined {
  let index = start;
  let prefixes = 0;
  for (;;) {
    const byte = bytes[index];
    if (byte === 0x66 || byte === 0x2e || byte === 0x36 || byte === 0x3e || byte === 0x26 || byte === 0x64 || byte === 0x65) {
      index += 1;
      prefixes += 1;
      continue;
    }
    break;
  }
  if (prefixes === 0) return undefined;
  if (bytes[index] !== 0x0f || bytes[index + 1] !== 0x1f) return undefined;
  return index + 2;
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

function registerOperand(code: number, extended: boolean, bits: 8 | 16 | 32 | 64, rexPresent = true): Operand {
  const index = code | (extended ? 8 : 0);
  if (bits === 8) return byteRegisterOperand(code, extended, rexPresent);
  if (bits === 16) return { kind: "reg", name: WORD_REGISTERS[index]!, bits };
  return { kind: "reg", name: REGISTERS[index]!, bits };
}

/**
 * A byte register named the way the encoding names it.
 *
 * Without REX, codes 4 to 7 are the high halves (`ah` and friends), and there is no way to spell `spl` to `dil`; with REX
 * they are the low bytes of the registers above `bl`. Both facts are in the encoding, so both are read from it rather
 * than assumed, which is the difference between `ah` and `spl` in a decoded function.
 */
function byteRegisterOperand(code: number, extended: boolean, rexPresent: boolean): Operand {
  const index = code | (extended ? 8 : 0);
  if (!rexPresent && index >= 4 && index <= 7) {
    return { kind: "reg", name: HIGH_BYTE_REGISTERS[index - 4]!, bits: 8 };
  }
  return { kind: "reg", name: BYTE_REGISTERS[index]!, bits: 8 };
}

interface ModRm {
  readonly mode: number;
  readonly regField: number;
  readonly rm: Operand;
}

function decodeModRm(
  cursor: Cursor,
  rex: Rex,
  bits: 8 | 32 | 64,
  segment?: "fs" | "gs" | "cs" | "ss" | "ds" | "es",
): ModRm {
  const byte = byteAt(cursor);
  const mode = byte >> 6;
  const regField = (byte >> 3) & 0x7;
  const rmField = byte & 0x7;

  if (mode === 3) return { mode, regField, rm: registerOperand(rmField, rex.b, bits, rex.present) };

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
  return { mode, regField, rm: segment === undefined ? memory : { ...memory, segment } };
}

// group 1 by /reg field: 0 add, 1 or, 2 adc, 3 sbb, 4 and, 5 sub, 6 xor, 7 cmp. The carry forms are left out on
// purpose: nothing a compiler emits for C code needs them, and a translator that ignored the carry would be wrong.
const ARITHMETIC: readonly (string | undefined)[] = ["add", "or", undefined, undefined, "and", "sub", "xor", "cmp"];
/** `jcc` and `setcc` share one condition order, which is the encoding's and not a coincidence. */
const CONDITIONS = [
  "o", "no", "b", "ae", "e", "ne", "be", "a",
  "s", "ns", "p", "np", "l", "ge", "le", "g",
] as const;
/** The shift group's `/reg` field: only the three that discard bits are in the subset. */
const SHIFTS: readonly (string | undefined)[] = [undefined, undefined, undefined, undefined, "shl", "shr", undefined, "sar"];

function decodeOne(bytes: Uint8Array, offset: number): Instruction {
  const cursor: Cursor = { bytes, start: offset, index: offset };
  let rex: Rex = { w: false, r: false, x: false, b: false, present: false };
  let segment: "fs" | "gs" | "cs" | "ss" | "ds" | "es" | undefined;

  for (;;) {
    const next = bytes[cursor.index];
    if (next === undefined) throw new DecodeError(offset, "the code buffer ends inside an instruction");
    if (next === 0x66) {
      /*
       * The one 0x66 sequence in the subset is the alignment padding a linker emits between functions: `66 2e 0f 1f 84 00`
       * and its shorter relatives, assembled by `as` as a multi-byte NOP. It is not a 16-bit operand here, and refusing it
       * stopped the decode of a real binary at the first function boundary rather than at real code.
       */
      const prefixes = alignmentNopPrefixes(bytes, cursor.index);
      if (prefixes !== undefined) {
        cursor.index = prefixes;
        // the nop's operand is an ordinary ModRM, and its width is 32 unless REX.W set it, which the nop ignores
        const modrm = decodeModRm(cursor, rex, 32, segment);
        return { offset, length: cursor.index - offset, op: "nop", operands: [modrm.rm] };
      }
      if (bytes[cursor.index + 1] === 0x0f) {
        // 0x66 0f is never a 16-bit operand: it is the mandatory prefix of an SSE encoding, and calling it a "16-bit
        // operand size prefix" in the error would point a reader at the wrong part of the architecture
        throw new DecodeError(
          offset,
          `0x66 0f 0x${(bytes[cursor.index + 2] ?? 0).toString(16)} is an SSE instruction, and SSE is not in the decoded subset`,
        );
      }
      throw new DecodeError(offset, "a 16-bit operand size prefix (0x66) is not in the decoded subset");
    }
    if (next === 0x2e || next === 0x36 || next === 0x3e || next === 0x26 || next === 0x64 || next === 0x65) {
      // cs, ss, ds, es, fs, gs in that order. The prefix is carried into the operand, not dropped: dropping `fs` would
      // turn a thread-local read into a read of the absolute address in the displacement
      segment =
        next === 0x2e ? "cs" : next === 0x36 ? "ss" : next === 0x3e ? "ds" : next === 0x26 ? "es"
          : next === 0x64 ? "fs" : "gs";
      cursor.index += 1;
      continue;
    }
    if (next === 0xf3 && bytes[cursor.index + 1] === 0x0f && bytes[cursor.index + 2] === 0x1e) {
      /*
       * `endbr64` (f3 0f 1e fa) is the only repeat prefix in the subset, and it is not a prefix here at all: it is the
       * function entry marker every modern toolchain emits, so a decoder that stops at it stops at the first instruction
       * of most real functions. It is decoded as its own instruction with no operands, and as a no-op by the translator.
       */
      cursor.index += 4;
      if (bytes[cursor.index - 1] !== 0xfa) throw new DecodeError(offset, "f3 0f 1e is not followed by fa: not endbr64");
      return { offset, length: 4, op: "endbr64", operands: [] };
    }
    if (next === 0xf0 || next === 0xf2 || next === 0xf3) {
      throw new DecodeError(offset, "a lock or repeat prefix is not in the decoded subset");
    }
    if (next >= 0x40 && next <= 0x4f) {
      cursor.index += 1;
      rex = {
        w: (next & 0x8) !== 0,
        r: (next & 0x4) !== 0,
        x: (next & 0x2) !== 0,
        b: (next & 0x1) !== 0,
        present: true,
      };
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
    if (second === 0x0b) return finish("ud2", []);
    if (second >= 0x80 && second <= 0x8f) {
      return finish(`j${CONDITIONS[second - 0x80]!}`, [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
    }
    if (second >= 0x90 && second <= 0x9f) {
      // setcc writes one byte, and the register in it follows the no-REX high-byte rule
      const modrm = decodeModRm(cursor, rex, 8, segment);
      return finish(`set${CONDITIONS[second - 0x90]!}`, [modrm.rm]);
    }
    if (second === 0xb6 || second === 0xbe) {
      // movzx / movsx from a byte: the source is a byte register or a byte of memory, the destination 32 or 64 bits
      const modrm = decodeModRm(cursor, rex, 8, segment);
      return finish(second === 0xb6 ? "movzx" : "movsx", [registerOperand(modrm.regField, rex.r, bits), modrm.rm]);
    }
    if (second === 0xb7 || second === 0xbf) {
      // a word either way. From memory the size lives in the opcode; from a register it would need the 0x66 prefix,
      // which is refused above, so a register here means the bytes are not what they claim
      const modrm = decodeModRm(cursor, rex, bits, segment);
      if (modrm.rm.kind === "reg") {
        throw new DecodeError(
          offset,
          "movzx/movsx from a 16-bit register needs a 0x66 prefix, which is not in the decoded subset",
        );
      }
      return finish(second === 0xb7 ? "movzx" : "movsx", [registerOperand(modrm.regField, rex.r, bits), modrm.rm]);
    }
    if (second === 0xaf) {
      const modrm = decodeModRm(cursor, rex, bits, segment);
      return finish("imul", [registerOperand(modrm.regField, rex.r, bits), modrm.rm]);
    }
    if (isSseOpcode(second)) {
      throw new DecodeError(
        offset,
        `0x0f 0x${second.toString(16)} is an SSE instruction, and SSE is not in the decoded subset`,
      );
    }
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
  if (opcode >= 0x70 && opcode <= 0x7f) {
    return finish(`j${CONDITIONS[opcode - 0x70]!}`, [{ kind: "immediate", value: readImmediate(cursor, 1) }]);
  }
  if (opcode === 0xc9) return finish("leave", []);
  if (opcode === 0xcc) return finish("int3", []);
  if (opcode === 0xc2) return finish("ret", [{ kind: "immediate", value: readImmediate(cursor, 2) }]);
  if (opcode === 0x98 || opcode === 0x99) {
    // 0x98 is cwde/cdqe and 0x99 is cdq/cqo depending on REX.W; without it the pair is the 32-bit one, which is what
    // `movl`-sized code uses and what the fixture exercises
    if (opcode === 0x98) return finish(rex.w ? "cdqe" : "cwde", []);
    return finish(rex.w ? "cqo" : "cdq", []);
  }
  if (opcode === 0x68) return finish("push", [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
  if (opcode === 0x6a) return finish("push", [{ kind: "immediate", value: readImmediate(cursor, 1) }]);
  if (opcode === 0xe8) return finish("call", [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
  if (opcode === 0xe9) return finish("jmp", [{ kind: "immediate", value: readImmediate(cursor, 4) }]);
  if (opcode === 0x8d) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    if (modrm.rm.kind !== "memory") {
      throw new DecodeError(
        offset,
        "lea with a register operand cannot be encoded, so these bytes are not what they claim",
      );
    }
    return finish("lea", [registerOperand(modrm.regField, rex.r, bits), modrm.rm]);
  }
  if (opcode === 0x88 || opcode === 0x8a) {
    // the byte moves, where the register's name comes from the encoding and not from a width suffix: `88 c4` is
    // `mov ah, al` and `40 88 c4` is `mov spl, al`
    const modrm = decodeModRm(cursor, rex, 8, segment);
    const register = registerOperand(modrm.regField, rex.r, 8, rex.present);
    return opcode === 0x88 ? finish("mov", [modrm.rm, register]) : finish("mov", [register, modrm.rm]);
  }
  if (opcode === 0x89 || opcode === 0x8b) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x89 ? finish("mov", [modrm.rm, register]) : finish("mov", [register, modrm.rm]);
  }
  if (opcode === 0x31 || opcode === 0x33) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x31 ? finish("xor", [modrm.rm, register]) : finish("xor", [register, modrm.rm]);
  }
  if (opcode === 0x01 || opcode === 0x29) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x01 ? finish("add", [modrm.rm, register]) : finish("sub", [modrm.rm, register]);
  }
  if (opcode === 0xc7) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    if (modrm.regField !== 0) {
      throw new DecodeError(offset, "0xc7 with a /reg field other than 0 is not a mov and is not in the decoded subset");
    }
    return finish("mov", [modrm.rm, { kind: "immediate", value: readImmediate(cursor, 4) }]);
  }
  if (opcode === 0x83 || opcode === 0x81) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const mnemonic = ARITHMETIC[modrm.regField];
    if (mnemonic === undefined) {
      throw new DecodeError(
        offset,
        `arithmetic /${modrm.regField} (0x${opcode.toString(16)}) is not in the decoded subset`,
      );
    }
    return finish(mnemonic, [modrm.rm, { kind: "immediate", value: readImmediate(cursor, opcode === 0x83 ? 1 : 4) }]);
  }

  if (opcode === 0x85) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    return finish("test", [modrm.rm, registerOperand(modrm.regField, rex.r, bits)]);
  }
  if (opcode === 0x63) {
    // movsxd, the sign-extending load a compiler emits for `int` to `long`. Only meaningful with REX.W.
    const modrm = decodeModRm(cursor, rex, bits, segment);
    return finish(rex.w ? "movsxd" : "arpl", [registerOperand(modrm.regField, rex.r, bits), modrm.rm]);
  }
  if (opcode === 0x69 || opcode === 0x6b) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const immediate = readImmediate(cursor, opcode === 0x69 ? 4 : 1);
    return finish("imul", [registerOperand(modrm.regField, rex.r, bits), modrm.rm, { kind: "immediate", value: immediate }]);
  }
  if (opcode === 0x39 || opcode === 0x3b) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x39 ? finish("cmp", [modrm.rm, register]) : finish("cmp", [register, modrm.rm]);
  }
  if (opcode === 0x03 || opcode === 0x2b) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x03 ? finish("add", [register, modrm.rm]) : finish("sub", [register, modrm.rm]);
  }
  if (opcode === 0x21 || opcode === 0x09) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const register = registerOperand(modrm.regField, rex.r, bits);
    return opcode === 0x21 ? finish("and", [modrm.rm, register]) : finish("or", [modrm.rm, register]);
  }
  if (opcode === 0xa9) {
    // test between the accumulator and an immediate: `test eax, imm32`, which is what `testl $255, %eax` assembles to
    return finish("test", [registerOperand(0, false, bits), { kind: "immediate", value: readImmediate(cursor, 4) }]);
  }
  if (opcode === 0x25 || opcode === 0x0d || opcode === 0x35 || opcode === 0x05 || opcode === 0x2d || opcode === 0x3d) {
    // the accumulator forms: and/or/xor/add/sub/cmp between eax (or rax) and an immediate, short and common
    const mnemonic =
      opcode === 0x25 ? "and" : opcode === 0x0d ? "or" : opcode === 0x35 ? "xor"
        : opcode === 0x05 ? "add" : opcode === 0x2d ? "sub" : "cmp";
    const accumulator = registerOperand(0, false, bits);
    return finish(mnemonic, [accumulator, { kind: "immediate", value: readImmediate(cursor, 4) }]);
  }
  if (opcode === 0xf7) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    if (modrm.regField === 0) {
      return finish("test", [modrm.rm, { kind: "immediate", value: readImmediate(cursor, 4) }]);
    }
    if (modrm.regField === 2) return finish("not", [modrm.rm]);
    if (modrm.regField === 3) return finish("neg", [modrm.rm]);
    throw new DecodeError(
      offset,
      `0xf7 /${modrm.regField} is mul, imul, div or idiv, which read and write implicit registers, and those are not in ` +
        "the decoded subset yet",
    );
  }
  if (opcode === 0xd1 || opcode === 0xd3 || opcode === 0xc1) {
    const modrm = decodeModRm(cursor, rex, bits, segment);
    const mnemonic = SHIFTS[modrm.regField];
    if (mnemonic === undefined) {
      throw new DecodeError(
        offset,
        `shift group /${modrm.regField} is a rotate or an undefined encoding, and neither is in the decoded subset`,
      );
    }
    if (opcode === 0xd1) return finish(mnemonic, [modrm.rm, { kind: "immediate", value: 1n }]);
    if (opcode === 0xd3) return finish(mnemonic, [modrm.rm, registerOperand(1, false, 8, rex.present)]);
    return finish(mnemonic, [modrm.rm, { kind: "immediate", value: readImmediate(cursor, 1) }]);
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
