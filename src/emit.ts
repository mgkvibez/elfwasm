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

/**
 * The memory a module addresses, as an import rather than something the module owns.
 *
 * Importing is what makes "memory against the mapped image" true: the caller builds `WebAssembly.Memory`, fills it with
 * the ELF's segments, and hands it in. A module that declared its own memory would have a buffer the loader never filled,
 * and every address would read zero while looking legitimate.
 */
export interface MemoryImport {
  /** Pages the caller promises, so a wrong-sized memory fails at instantiation instead of at a wild read. */
  readonly minimumPages: number;
}

export interface ModuleOptions {
  /** Export name, so a test can call the translated function. */
  readonly name: string;
  /** Locals the bodies index: the translator's own scratch slots, each an i64. */
  readonly locals: number;
  /**
   * One body per function, in index order: index 0 is the exported entry, and a `call` names one of the others.
   *
   * Every function has the same signature, no parameters and one `i64` result, which is `%rax`. That is not a shortcut: the
   * machine's argument registers are module globals, so a call passes its arguments by writing them, exactly as SysV says,
   * and the result comes back as the return value.
   */
  readonly bodies: readonly (readonly number[])[];
  /** How many `i64` globals to declare, which is where the register file lives. */
  readonly globals?: number;
  /** Present only when the body touches memory, so a module that reads nothing keeps needing nothing. */
  readonly memory?: MemoryImport;
}

/**
 * Builds the module: type, import, function, export, then code, in the order the format requires.
 *
 * There is at most one import, and it does not occupy the function index space, so the export index stays equal to the
 * function index when memory is imported and when it is not. That is the off-by-one this file cannot make.
 */
export function module(options: ModuleOptions): Uint8Array {
  // the type section's body is a vector: a count, then the signature itself
  const typeBody = [0x01, 0x60, 0x00, 0x01, I64]; // one type: no params, one i64 result
  // every function shares that one type, so the function section is a count and a run of zeroes
  const functionBody = [...unsigned(options.bodies.length), ...options.bodies.map(() => 0x00)];

  /*
   * The global section, which is where the sixteen registers live. Each is mutable and starts at zero, which is the state a
   * process begins in. They are globals rather than locals because a register outlives a call: a callee that pushes `%rbp`
   * has to push the caller's `%rbp`, and a local per function would push zero instead.
   */
  const globalSection =
    options.globals === undefined || options.globals === 0
      ? []
      : section(6, [
          ...unsigned(options.globals),
          ...Array.from({ length: options.globals }, () => [I64, 0x01, 0x42, 0x00, 0x0b]).flat(),
        ]);

  /*
   * The import section, written only when there is something to import. Module and field names are the conventional
   * `env.memory`; the descriptor byte 0x02 says "memory", 0x00 says "limits with a minimum and no maximum", and the
   * minimum is a count of 64 KiB pages rather than a byte length.
   */
  const importSection =
    options.memory === undefined
      ? []
      : section(2, [
          0x01, // one import
          0x03, 0x65, 0x6e, 0x76, // module name "env"
          0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, // field name "memory"
          0x02, // a memory
          0x00, // limits: minimum only
          ...unsigned(options.memory.minimumPages),
        ]);
  const exportName = [...options.name].map((character) => character.charCodeAt(0));
  const exportBody = [0x01, ...unsigned(exportName.length), ...exportName, 0x00, 0x00]; // one export, of function 0

  /*
   * One local group of the same type, which is all a register file needs, then the body and its `end`. The size that
   * precedes it is computed from the bytes rather than written by hand, so a body that grows cannot leave a stale length
   * behind: that is the one error in this format that produces a module which loads and misbehaves rather than one that
   * fails to load.
   */
  const functions = options.bodies.map((body) => {
    const instructions = [0x01, ...unsigned(options.locals), I64, ...body, 0x0b];
    return [...unsigned(instructions.length), ...instructions];
  });
  const codeBody = [...unsigned(functions.length), ...functions.flat()];

  return Uint8Array.from([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...section(1, typeBody),
    ...importSection,
    ...section(3, functionBody),
    ...globalSection,
    ...section(7, exportBody),
    ...section(10, codeBody),
  ]);
}

/** Instruction encodings this emitter uses, named so the translator does not carry bare hex bytes. */
export const OP = {
  call: 0x10,
  globalGet: 0x23,
  globalSet: 0x24,
  localGet: 0x20,
  localSet: 0x21,
  localTee: 0x22,
  i32Const: 0x41,
  i64Const: 0x42,
  i64Add: 0x7c,
  i64Sub: 0x7d,
  i64Mul: 0x7e,
  i64Xor: 0x85,
  i64And: 0x83,
  i64Or: 0x84,
  i64Shl: 0x86,
  i32WrapI64: 0xa7,
  i64ExtendI32U: 0xad,
  // a trap, which is what an address outside the image becomes: a wild read is worse than a stopped program
  unreachable: 0x00,
  // memory access: the address is on the stack and the memarg carries alignment and a constant offset
  i32Load: 0x28,
  i64Load: 0x29,
  i32Store: 0x36,
  i64Store: 0x37,
  i32Store8: 0x3a,
  i32Store16: 0x3b,
  // control flow, which is what a branch translates into rather than into jumps
  i32Eqz: 0x45,
  ifEmpty: 0x04,
  elseOp: 0x05,
  emptyBlockType: 0x40,
  block: 0x02,
  loop: 0x03,
  br: 0x0c,
  brIf: 0x0d,
  returnOp: 0x0f,
  end: 0x0b,
  // comparisons, which are how the flags become a condition: the flags themselves are never materialised
  i32Eq: 0x46, i32Ne: 0x47, i32LtS: 0x48, i32LtU: 0x49, i32GtS: 0x4a,
  i32GtU: 0x4b, i32LeS: 0x4c, i32LeU: 0x4d, i32GeS: 0x4e, i32GeU: 0x4f,
  i64Eq: 0x51, i64Ne: 0x52, i64LtS: 0x53, i64LtU: 0x54, i64GtS: 0x55,
  i64GtU: 0x56, i64LeS: 0x57, i64LeU: 0x58, i64GeS: 0x59, i64GeU: 0x5a,
  i32Add: 0x6a,
  i32Sub: 0x6b,
  i32Mul: 0x6c,
  i32And: 0x71,
  i32Or: 0x72,
  i32Xor: 0x73,
} as const;

export function localGet(index: number): number[] {
  return [OP.localGet, ...unsigned(index)];
}

export function localSet(index: number): number[] {
  return [OP.localSet, ...unsigned(index)];
}

export function globalGet(index: number): number[] {
  return [OP.globalGet, ...unsigned(index)];
}

export function globalSet(index: number): number[] {
  return [OP.globalSet, ...unsigned(index)];
}

export function i64Const(value: bigint): number[] {
  return [OP.i64Const, ...signed64(value)];
}

/**
 * An i32 constant, with the value folded into 32 bits before it is encoded.
 *
 * The immediate is a *signed* LEB, so `0xffffffff` has to go out as `-1`: encoding the unsigned number needs a fifth byte
 * with bits above 32 set, and the engine rejects the module with "extra bits in varint". Folding here means a caller that
 * thinks in unsigned hex still gets a module that loads.
 */
export function i32Const(value: number): number[] {
  return [OP.i32Const, ...signed64(BigInt.asIntN(32, BigInt(value)))];
}

/**
 * A memory access's immediate pair: alignment as a power-of-two exponent, then a constant offset.
 *
 * Alignment 0 (one byte) is used everywhere: WASM permits unaligned access, and claiming a stronger alignment than the
 * address holds is undefined behaviour in the specification rather than something the engine checks for you.
 */
export function memarg(offset: number): number[] {
  return [...unsigned(0), ...unsigned(offset)];
}

/** Unused by the module writer above, but part of the public surface a caller assembling several sections needs. */
export { section, vector };
