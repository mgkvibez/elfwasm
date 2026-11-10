/**
 * A WASM module writer: the sections a translated function needs, in the order the binary format requires.
 *
 * This is deliberately not a general WASM library. It emits one function with no parameters and one i64 result, because
 * that is the shape the translator produces first, and a builder that only knows that shape cannot quietly emit a module
 * that disagrees with itself. Everything a caller needs is checked: a section body's length is computed from the bytes it
 * holds, so a wrong length is not expressible, and a module is only produced through `finish`, which writes the type,
 * function, export and code sections in that order.
 *
 * Values below are the ones the format fixes, written as words rather than as bare numbers so a reader can check them
 * against the specification instead of against this file.
 */

/** Value types, as the format encodes them in a function's signature and in a local declaration. */
export const I32 = 0x7f;
export const I64 = 0x7e;

/** Unsigned LEB128, the encoding every length, count and index uses. */
export function unsigned(value: number): number[] {
  const bytes: number[] = [];
  let rest = value;
  do {
    let byte = rest & 0x7f;
    rest >>>= 7;
    if (rest !== 0) byte |= 0x80;
    bytes.push(byte);
  } while (rest !== 0);
  return bytes;
}

/**
 * Signed LEB128, which `i64.const` needs.
 *
 * The last byte must carry the value's sign, or the decoder reads a different number: `-1` is one byte, and `64` is two,
 * because a single byte of `0x40` would decode as `-64`.
 */
export function signed64(value: bigint): number[] {
  const bytes: number[] = [];
  let rest = value;
  for (;;) {
    const byte = Number(rest & 0x7fn);
    rest >>= 7n;
    const signBitSet = (byte & 0x40) !== 0;
    if ((rest === 0n && !signBitSet) || (rest === -1n && signBitSet)) {
      bytes.push(byte);
      return bytes;
    }
    bytes.push(byte | 0x80);
  }
}

/** A section with its id, its body, and the body's length written from the body itself. */
function section(id: number, body: readonly number[]): number[] {
  return [id, ...unsigned(body.length), ...body];
}

function vector<T>(items: readonly T[], encode: (item: T) => number[]): number[] {
  return [...unsigned(items.length), ...items.flatMap(encode)];
}

export interface ModuleOptions {
  /** Export name, so a test can call the translated function. */
  readonly name: string;
  /** Locals the body indexes, each an i64 holding one x86-64 register. */
  readonly locals: number;
  /** The body's instructions, without the terminating `end`, which `finish` writes. */
  readonly body: readonly number[];
  /** Index of the only result, which the body must leave on the stack. */
  readonly result: typeof I32 | typeof I64;
}

/**
 * Builds the module: type, function, export, then code, in the order the format requires.
 *
 * There is one function and no imports. That keeps the export index equal to the function index, so an off-by-one in one
 * of them is not a mistake this file can make.
 */
export function module(options: ModuleOptions): Uint8Array {
  // the type section's body is a vector: a count, then the signature itself
  const typeBody = [0x01, 0x60, 0x00, 0x01, options.result]; // one type: no params, one result
  const functionBody = [0x01, 0x00]; // one function, of type 0
  const exportName = [...options.name].map((character) => character.charCodeAt(0));
  const exportBody = [0x01, ...unsigned(exportName.length), ...exportName, 0x00, 0x00]; // one export, of function 0

  /*
   * One local group of the same type, which is all a register file needs, then the body and its `end`. The size that
   * precedes it is computed from the bytes rather than written by hand, so a body that grows cannot leave a stale length
   * behind: that is the one error in this format that produces a module which loads and misbehaves rather than one that
   * fails to load.
   */
  const instructions = [0x01, ...unsigned(options.locals), I64, ...options.body, 0x0b];
  const codeBody = [0x01, ...unsigned(instructions.length), ...instructions];

  return Uint8Array.from([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...section(1, typeBody),
    ...section(3, functionBody),
    ...section(7, exportBody),
    ...section(10, codeBody),
  ]);
}

/** Instruction encodings this emitter uses, named so the translator does not carry bare hex bytes. */
export const OP = {
  localGet: 0x20,
  localSet: 0x21,
  i32Const: 0x41,
  i64Const: 0x42,
  i64Add: 0x7c,
  i64Sub: 0x7d,
  i64Mul: 0x7e,
  i64Xor: 0x85,
  i64And: 0x83,
  i64Or: 0x84,
  i64Shl: 0x86,
  i64Eq: 0x51,
  i64Ne: 0x52,
  i64LtS: 0x53,
  i64GtS: 0x55,
  i64LeS: 0x57,
  i64GeS: 0x59,
  i32WrapI64: 0xa7,
  i64ExtendI32U: 0xad,
} as const;

export function localGet(index: number): number[] {
  return [OP.localGet, ...unsigned(index)];
}

export function localSet(index: number): number[] {
  return [OP.localSet, ...unsigned(index)];
}

export function i64Const(value: bigint): number[] {
  return [OP.i64Const, ...signed64(value)];
}

export function i32Const(value: number): number[] {
  return [OP.i32Const, ...signed64(BigInt(value))];
}

/** Unused by the module writer above, but part of the public surface a caller assembling several sections needs. */
export { section, vector };
