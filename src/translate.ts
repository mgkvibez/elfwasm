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

import { decode, type Instruction, type Operand } from "./decode.js";
import { I64, OP, i32Const, i64Const, localGet, localSet, memarg, module, unsigned } from "./emit.js";
import type { AddressSpace } from "./image.js";

/** The register file, in the order the encoding numbers it. Index is the WASM local. */
const REGISTERS = [
  "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
] as const;

/** The two locals `cmp` and `test` leave their operands in, so a branch can be the comparison. */
const CMP_A = REGISTERS.length;
const CMP_B = REGISTERS.length + 1;
/**
 * Where a computed address waits between its bound check and its use.
 *
 * It cannot be one of the comparison locals: `cmp (%rax), %rbx` keeps its first operand in `CMP_A` while the second is being
 * loaded, and an address check that wrote `CMP_A` there would silently compare the address against the value.
 */
const ADDR = REGISTERS.length + 2;
const LOCAL_COUNT = REGISTERS.length + 3;

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
  /** Virtual address of the first byte of `instructions`, for RIP-relative operands. Defaults to 0. */
  readonly codeAddress?: bigint;
  /** Pages to import when the body touches memory. Defaults to the image's own pages plus `stackPages`. */
  readonly memoryPages?: number;
  /**
   * Pages above the image reserved for the stack, and the opt-in that enables stack translation at all.
   *
   * It is not a default because it changes what the caller has to allocate: the import minimum grows, so a caller that
   * supplies the image's pages alone gets an instantiation error rather than a program with a stack it did not ask for.
   */
  readonly stackPages?: number;
}

export class TranslateError extends Error {
  readonly offset: number;

  constructor(offset: number, message: string) {
    super(`at offset 0x${offset.toString(16)}: ${message}`);
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

function localOf(operand: Operand, instruction: Instruction): number {
  if (operand.kind !== "reg") {
    throw new TranslateError(instruction.offset, `expected a register operand, found ${operand.kind}`);
  }
  return localOfRegister(operand.name, instruction);
}

function read(operand: Operand, instruction: Instruction): number[] {
  return localGet(localOf(operand, instruction));
}

function write(operand: Operand, instruction: Instruction): number[] {
  return localSet(localOf(operand, instruction));
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
    return (options.codeAddress ?? 0n) + BigInt(instruction.offset + instruction.length) + operand.displacement;
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
  if (operand.segment !== undefined) {
    throw new TranslateError(
      instruction.offset,
      `a %${operand.segment}: operand needs a segment base, and WASM has no segments: refusing rather than reading the ` +
        `offset as a plain address`,
    );
  }
  // the address is a sum of parts, and each part is pushed before the add that consumes the two before it
  const parts: number[][] = [];
  if (operand.base !== undefined) {
    // a register's value is a virtual address, so the image's own base is subtracted before anything is addressed
    parts.push([
      ...localGet(localOfRegister(operand.base, instruction)),
      ...i64Const(addresses.baseAddress),
      OP.i64Sub,
    ]);
  }
  if (operand.index !== undefined) {
    const shift = { 1: 0n, 2: 1n, 4: 2n, 8: 3n }[operand.scale];
    const scaled = [...localGet(localOfRegister(operand.index, instruction))];
    if (shift !== 0n) scaled.push(...i64Const(shift), OP.i64Shl);
    parts.push(scaled);
  }
  const code = [...parts[0]!];
  for (const part of parts.slice(1)) code.push(...part, OP.i64Add);
  if (operand.displacement !== 0n) code.push(...i64Const(operand.displacement), OP.i64Add);

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
        if (operand.base !== undefined || operand.index !== undefined) {
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
  return [...localGet(rsp), ...i64Const(bytes), OP.i64Add, ...localSet(rsp)];
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
    ...localGet(REGISTERS.indexOf("rsp")),
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
    ...localGet(rbp),
    ...localSet(REGISTERS.indexOf("rsp")),
    // the pop reads through the same address machinery as any other stack access, so it is bound-checked too
    ...runtimeAddressCode(STACK_OPERAND, instruction, options, 64),
    OP.i64Load,
    ...memarg(0),
    ...localSet(rbp),
    ...adjustStack(8n),
  ];
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
function loadFromMemory(operand: Operand, width: 32 | 64, instruction: Instruction): number[] {
  const address = addressOf(operand, instruction);
  if (width === 64) return [...address, OP.i64Load, ...memarg(0)];
  // a 32-bit load zeroes the upper half of the destination, which is x86's own rule for `movl`
  return [...address, OP.i32Load, ...memarg(0), WIDEN_32];
}

/** A store: the address goes on the stack first, the value on top, which is the order `store` pops them in. */
function storeToMemory(operand: Operand, value: readonly number[], width: 8 | 16 | 32 | 64, instruction: Instruction): number[] {
  const code = { 8: OP.i32Store8, 16: OP.i32Store16, 32: OP.i32Store, 64: OP.i64Store }[width];
  return [...addressOf(operand, instruction), ...value, code, ...memarg(0)];
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
  if (width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit mov is not translated yet`);
  }
  if (source.kind === "immediate") {
    // a 32-bit destination zero-extends, so this masks rather than sign-extends: `mov $-1, %eax` leaves 0xffffffff
    const value = width === 32 ? BigInt.asUintN(32, source.value) : source.value;
    return [...i64Const(value), ...write(destination, instruction)];
  }
  if (source.kind === "memory") {
    // the load already yields the destination's width, so no narrowing of a register is involved
    return [...loadFromMemory(source, width, instruction), ...write(destination, instruction)];
  }
  const code = read(source, instruction);
  if (width === 32) return [...code, NARROW_32, WIDEN_32, ...write(destination, instruction)];
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
    const width = operand.bits === 32 ? 32 : 64;
    return loadFromMemory(operand, width, instruction);
  }
  return read(operand, instruction);
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
  const operand = operandCode(source, width, instruction);
  const opcodes = BINARY[op];
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
function translateFlags(instruction: Instruction): { code: number[]; width: 32 | 64 } {
  const [destination, source] = instruction.operands;
  if (destination === undefined || source === undefined) {
    throw new TranslateError(instruction.offset, `${instruction.op} needs two operands`);
  }
  if (destination.kind !== "reg") {
    throw new TranslateError(instruction.offset, `${instruction.op} into memory is not translated yet`);
  }
  const width = destination.bits;
  if (width !== 32 && width !== 64) {
    throw new TranslateError(instruction.offset, `a ${width}-bit ${instruction.op} is not translated yet`);
  }
  const left = read(destination, instruction);
  if (instruction.op === "test") {
    // `test a, b` is `and` with the result discarded: the flags come from a & b, and comparing the mask against zero
    // reproduces every condition a compiler emits after it, jl and js included
    const mask = [...left, ...operandCode64(source, instruction), OP.i64And];
    return {
      code: [...mask, ...localSet(CMP_A), ...i64Const(0n), ...localSet(CMP_B)],
      width,
    };
  }
  return {
    code: [...left, ...localSet(CMP_A), ...operandCode64(source, instruction), ...localSet(CMP_B)],
    width,
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

/** Both operands of the last comparison, at the width it was made, ready for one comparison opcode. */
function comparisonOperands(width: 32 | 64): number[] {
  if (width === 64) return [...localGet(CMP_A), ...localGet(CMP_B)];
  return [...localGet(CMP_A), NARROW_32, ...localGet(CMP_B), NARROW_32];
}

/**
 * A condition as an i32 on the stack.
 *
 * `js` and `jns` are the only two that need the subtraction rather than a comparison, because the sign flag is the sign
 * of the difference: `a - b` is pushed and compared against zero, which is the definition rather than a shortcut.
 */
function conditionCode(suffix: string, width: 32 | 64): number[] {
  if (suffix === "s" || suffix === "ns") {
    const difference =
      width === 64
        ? [...comparisonOperands(64), OP.i64Sub, ...i64Const(0n), OP.i64LtS]
        : [...comparisonOperands(32), OP.i32Sub, ...i32Const(0), OP.i32LtS];
    return suffix === "s" ? difference : [...difference, OP.i32Eqz];
  }
  const comparison = COMPARISONS[suffix as keyof typeof COMPARISONS];
  if (comparison === undefined) {
    throw new Error(`unreachable: ${suffix} has no comparison`);
  }
  return [...comparisonOperands(width), COMPARE_OPCODES[width][comparison]];
}

/** A region's exit points: the instruction index of its end, and how many blocks a `br` must cross to reach it. */
type Scope = Map<number, number>;

interface Flags {
  set: boolean;
  width: 32 | 64;
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
    if (instruction.op === "nop" || instruction.op === "endbr64") {
      index += 1;
      continue;
    }
    if (instruction.op === "ret") {
      if (instruction.operands.some((operand) => operand.kind === "immediate")) {
        throw new TranslateError(instruction.offset, "ret with a stack adjustment is not translated yet");
      }
      // `return` rather than a value left on the stack: a ret inside a branch cannot leave the function's result there
      out.push(...localGet(REGISTERS.indexOf("rax")), OP.returnOp);
      index += 1;
      if (index < to) {
        throw new TranslateError(instructions[index]!.offset, "there is code after ret, which is dead and not translated");
      }
      return index;
    }
    if (instruction.op === "jmp") {
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
        throw new TranslateError(instruction.offset, `${instruction.op} needs a cmp or test before it in the same block`);
      }
      const target = indexOfOffset(instructions, branchTarget(instruction));
      const condition = conditionCode(suffix, flags.width);
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
        out.push(...condition, OP.ifEmpty, OP.emptyBlockType);
        translateRange(instructions, target, join.join, new Map(shared), out, { set: false, width: 64 }, loops, options);
        out.push(OP.elseOp);
        translateRange(instructions, index + 1, join.jmp, new Map(shared), out, { set: false, width: 64 }, loops, options);
        out.push(OP.end);
        flags.set = false;
        index = join.join;
        continue;
      }
      // a jump forward inside this region: an `if` around the bytes it skips, with the condition negated
      out.push(...condition, OP.i32Eqz, OP.ifEmpty, OP.emptyBlockType);
      const nested: Scope = new Map([...scope].map(([key, value]) => [key, value + 1] as const));
      nested.set(target, 0);
      translateRange(instructions, index + 1, target, nested, out, { set: flags.set, width: flags.width }, loops, options);
      out.push(OP.end);
      // after the join, a comparison made on one side is not known to have happened on the other, so the flags are
      // dropped rather than assumed
      flags.set = false;
      index = target;
      continue;
    }
    if (instruction.op === "cmp" || instruction.op === "test") {
      const translated = translateFlags(instruction);
      out.push(...translated.code);
      flags.set = true;
      flags.width = translated.width;
      index += 1;
      continue;
    }
    if (instruction.op === "push" || instruction.op === "pop") {
      out.push(
        ...(instruction.op === "push"
          ? translatePush(instruction, options)
          : translatePop(instruction, options)),
      );
      // a push or pop touches memory, so a condition set before it cannot be trusted after it
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "leave") {
      out.push(...translateLeave(instruction, options));
      flags.set = false;
      index += 1;
      continue;
    }
    if (instruction.op === "call") {
      throw new TranslateError(
        instruction.offset,
        "call is not translated yet: a translated function has no calling convention to call another with",
      );
    }
    if (instruction.op === "mov") {
      out.push(...translateMov(instruction));
      index += 1;
      continue;
    }
    if (instruction.op in BINARY) {
      out.push(...translateBinary(instruction));
      index += 1;
      continue;
    }
    throw new TranslateError(instruction.offset, `${instruction.op} is not translated yet`);
  }
  return index;
}

/**
 * Translates one function.
 *
 * The function must end in `ret`, and the value it returns is `rax`, which is the only calling convention this increment
 * claims. Loops, jumps out of the branch they belong to, and anything the instruction set covers but the translator does
 * not are refused by name and offset.
 */
export function translate(
  instructions: readonly Instruction[],
  name = "main",
  options: TranslateOptions = {},
): Uint8Array {
  if (instructions.length === 0) {
    throw new TranslateError(0, "there is nothing to translate");
  }
  instructions = resolveAddresses(instructions, options);
  const body: number[] = [];
  const flags: Flags = { set: false, width: 64 };
  const loops = findLoops(instructions);
  /*
   * The stack's pointer is set here rather than by the caller, because `%rsp` is a local and locals start at zero. Pointing
   * it at the top of the imported memory is the one decision that makes every later push and pop arithmetic rather than a
   * special case, and it happens before the body so a stack access in the first instruction is already legal.
   */
  if (usesStack(instructions)) {
    if ((options.stackPages ?? 0) < 1) {
      throw new TranslateError(
        0,
        "this function uses the stack, so it needs stackPages: pass how many pages to set aside above the image",
      );
    }
    const { top } = stackBounds(options);
    body.push(...i64Const(top), ...localSet(REGISTERS.indexOf("rsp")));
  }
  translateRange(instructions, 0, instructions.length, new Map(), body, flags, loops, options);
  if (!instructions.some((instruction) => instruction.op === "ret")) {
    throw new TranslateError(
      instructions.at(-1)?.offset ?? 0,
      "the block does not end in ret, so it has no value to return",
    );
  }
  const stackUsed = usesStack(instructions);
  const touchesMemory =
    stackUsed ||
    instructions.some((instruction) => instruction.operands.some((operand) => operand.kind === "memory"));
  return module({
    name,
    locals: LOCAL_COUNT,
    body,
    result: I64,
    // the import appears only when something addresses memory, so a module that reads nothing still needs nothing
    ...(touchesMemory || stackUsed ? { memory: { minimumPages: totalPages(options) } } : {}),
  });
}

/** Bytes to a module, which is the path every test uses: decode, translate, run. */
export function translateBytes(bytes: Uint8Array, name = "main", options: TranslateOptions = {}): Uint8Array {
  return translate(decode(bytes), name, options);
}
