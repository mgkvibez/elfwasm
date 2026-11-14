import { describe, expect, it } from "vitest";

import { decode } from "../src/decode.js";
import { signed64, unsigned } from "../src/emit.js";
import { TranslateError, translate, translateBytes } from "../src/translate.js";

const hex = (text: string): Uint8Array =>
  Uint8Array.from(text.trim().split(/\s+/).map((pair) => Number.parseInt(pair, 16)));

/**
 * Runs a translated module and returns what it computes.
 *
 * The point of every test here: the number is produced by WebAssembly from bytes that began life as x86-64, and it is
 * compared against a value this file states. A translator that agreed with its own expectations would prove nothing.
 */
/**
 * The runtime, reached through `globalThis` with the smallest surface this file needs.
 *
 * The project has no `dom` lib and no `@types/node` on purpose, so `WebAssembly` is not in the type surface. Typing only
 * the two constructors used keeps that dependency-free arrangement while still compiling, and it is honest about how
 * little of the runtime this test relies on.
 */
const runtime = (
  globalThis as unknown as {
    WebAssembly: {
      Module: new (bytes: Uint8Array) => unknown;
      Instance: new (module: unknown) => { exports: Record<string, unknown> };
    };
  }
).WebAssembly;

const run = (bytes: Uint8Array): bigint => {
  const instance = new runtime.Instance(new runtime.Module(translateBytes(bytes)));
  const entry = instance.exports.main as () => bigint;
  return entry();
};

describe("the LEB128 encodings", () => {
  it("writes unsigned lengths, including the multi-byte ones", () => {
    expect(unsigned(0)).toEqual([0x00]);
    expect(unsigned(127)).toEqual([0x7f]);
    expect(unsigned(128)).toEqual([0x80, 0x01]);
    expect(unsigned(624485)).toEqual([0xe5, 0x8e, 0x26]);
  });

  it("writes signed values whose last byte carries the sign", () => {
    // 64 needs two bytes: a lone 0x40 would decode as -64, which is the mistake this encoding invites
    expect(signed64(64n)).toEqual([0xc0, 0x00]);
    expect(signed64(-1n)).toEqual([0x7f]);
    expect(signed64(-64n)).toEqual([0x40]);
    expect(signed64(127n)).toEqual([0xff, 0x00]);
  });
});

describe("translating straight-line code", () => {
  it("adds an immediate and returns the register, which is rax by the convention this claims", () => {
    // mov $40, %rax ; add $2, %rax ; ret
    expect(run(hex("48 c7 c0 28 00 00 00 48 83 c0 02 c3"))).toBe(42n);
  });

  it("computes with registers as well as immediates", () => {
    // mov $7, %rax ; mov $6, %rcx ; imul %rcx, %rax ; ret
    expect(run(hex("48 c7 c0 07 00 00 00 48 c7 c1 06 00 00 00 48 0f af c1 c3"))).toBe(42n);
  });

  it("zeroes the upper half on a 32-bit write, which is what makes mov $-1, %eax positive", () => {
    // mov $-1, %eax ; ret  ->  rax is 0x00000000ffffffff
    expect(run(hex("b8 ff ff ff ff c3"))).toBe(0xffffffffn);
  });

  it("does 32-bit arithmetic at 32 bits, so the result wraps instead of widening", () => {
    // mov $-1, %eax ; add $2, %eax ; ret  ->  1, not 0x100000001
    expect(run(hex("b8 ff ff ff ff 83 c0 02 c3"))).toBe(1n);
  });

  it("keeps 64-bit arithmetic at 64 bits", () => {
    // mov $-1, %rax ; add $2, %rax ; ret
    expect(run(hex("48 c7 c0 ff ff ff ff 48 83 c0 02 c3"))).toBe(1n);
  });

  it("translates the same bytes twice to the same module, so the emitter has no hidden state", () => {
    const bytes = hex("48 c7 c0 28 00 00 00 c3");
    expect([...translateBytes(bytes)]).toEqual([...translateBytes(bytes)]);
  });

  it("skips the instructions that exist only for the linker and the branch predictor", () => {
    // endbr64 ; nop ; mov $5, %rax ; ret
    expect(run(hex("f3 0f 1e fa 90 48 c7 c0 05 00 00 00 c3"))).toBe(5n);
  });
});

describe("refusals", () => {

  it("refuses a condition that reads the bit a shift dropped, which is not the shifted value", () => {
    // mov $0x80000000,%eax ; sar $31,%eax ; jb done ; ret ; done: ret
    expect(() => run(hex("b8 00 00 00 80 c1 f8 1f 72 01 c3 c3"))).toThrow(/shift|carry|modelled/);
  });
  const refuse = (bytes: Uint8Array): TranslateError => {
    try {
      translateBytes(bytes);
    } catch (error) {
      if (error instanceof TranslateError) return error;
      throw error;
    }
    throw new Error("expected the translator to refuse these bytes");
  };

  it("refuses a pcmpeqd whose source is memory, since sixteen bytes are two loads", () => {
    expect(() => run(hex("66 0f 76 45 00 c3"))).toThrow(/pcmpeqd from memory/);
  });

  it("refuses an adc with no comparison before it, since there is no carry to add", () => {
    expect(refuse(hex("48 83 d1 00 c3")).message).toContain("adc reads the carry");
  });

  it("refuses a branch with no comparison before it, rather than deciding what the flags were", () => {
    expect(refuse(hex("48 c7 c0 01 00 00 00 75 00 c3")).message).toContain("needs a cmp or test");
  });

  it("refuses the two flags it does not compute, overflow and parity", () => {
    expect(refuse(hex("48 85 c0 70 00 c3")).message).toContain("overflow or parity");
  });

  it("drops the comparison after a join, so the next branch cannot rely on flags one path already changed", () => {
    // mov $1,%eax ; cmp $1,%eax ; je +0 ; je +0 ; ret
    expect(refuse(hex("b8 01 00 00 00 83 f8 01 74 00 74 00 c3")).message).toContain("needs a cmp or test");
  });

  it("refuses memory without the mapped image, which is what an address needs to become an offset", () => {
    // both forms are translated now; what is refused is doing it without an address space to resolve against
    expect(refuse(hex("48 8b 07 c3")).message).toContain("needs the mapped image");
    expect(refuse(hex("48 89 07 c3")).message).toContain("needs the mapped image");
    expect(refuse(hex("48 8b 04 25 00 01 40 00 c3")).message).toContain("needs the mapped image");
  });

  it("refuses a syscall unless the syscall layer was asked for", () => {
    expect(refuse(hex("48 c7 c0 01 00 00 00 0f 05 c3")).message).toContain("syscall needs the syscall layer");
  });

  it("refuses a block that does not return, rather than emitting a function with nothing on the stack", () => {
    expect(refuse(hex("48 c7 c0 01 00 00 00")).message).toContain("does not end in ret");
  });

  it("translates the bytes after a ret, which a branch elsewhere may reach", () => {
    /*
     * `mov $1,%rax; ret; ret`. The second ret is unreachable along this path and is translated in place rather than refused: a body
     * with more than one exit has blocks after a ret in it, and a branch into one of them is what a compiled resolver looks like.
     */
    expect(translate(decode(hex("48 c7 c0 01 00 00 00 c3 c3"))).length).toBeGreaterThan(0);
  });

  it("reports the offset of the instruction it refused, so a failure is locatable in the binary", () => {
    // the syscall is the third instruction: 7 bytes for the mov, 4 for the add, so 11 bytes in
    expect(refuse(hex("48 c7 c0 01 00 00 00 48 83 c0 01 0f 05 c3")).offset).toBe(11);
  });
});

describe("the translator and the decoder agree about a real listing", () => {
  it("translates decoded bytes rather than raw ones, so the two stages share one reading of the encoding", () => {
    const bytes = hex("48 c7 c0 2a 00 00 00 c3");
    const instructions = decode(bytes);
    expect(translate(instructions).length).toBeGreaterThan(0);
    expect([...translate(instructions)]).toEqual([...translateBytes(bytes)]);
  });
});

describe("flags become comparisons, not flag variables", () => {
  it("takes a conditional branch when the comparison holds, skipping the bytes under it", () => {
    // mov $5,%rax ; cmp $5,%rax ; jne +7 ; mov $1,%rax ; ret
    expect(run(hex("48 c7 c0 05 00 00 00 48 83 f8 05 75 07 48 c7 c0 01 00 00 00 c3"))).toBe(1n);
  });

  it("falls through the same bytes when it does not hold", () => {
    // the identical program with 6 against 5, so the jne is taken and rax keeps 6
    expect(run(hex("48 c7 c0 06 00 00 00 48 83 f8 05 75 07 48 c7 c0 01 00 00 00 c3"))).toBe(6n);
  });

  it("compares at 32 bits when the operands are 32-bit, so -1 is less than 1 after mov $-1, %eax", () => {
    // mov $-1,%eax ; cmp $1,%eax ; jl +7 ; mov $2,%rax ; ret
    expect(run(hex("b8 ff ff ff ff 83 f8 01 7c 07 48 c7 c0 02 00 00 00 c3"))).toBe(0xffffffffn);
  });

  it("compares at 64 bits when the operands are 64-bit, where -1 is also less than 1", () => {
    // WebAssembly's i64 arrives in JS as a signed BigInt, so the all-ones result is -1 here
    expect(run(hex("48 c7 c0 ff ff ff ff 48 83 f8 01 7c 07 48 c7 c0 02 00 00 00 c3"))).toBe(-1n);
  });

  it("reads jb as unsigned at 32 bits, where 0xffffffff is not below 1", () => {
    // mov $-1,%eax ; cmp $1,%eax ; jb +7 ; mov $2,%rax ; ret  ->  the branch is not taken
    expect(run(hex("b8 ff ff ff ff 83 f8 01 72 07 48 c7 c0 02 00 00 00 c3"))).toBe(2n);
  });

  it("reads js after test as the sign of the mask, which is the idiom a compiler emits for a negative int", () => {
    // mov $-1,%eax ; test %eax,%eax ; js +7 ; mov $2,%rax ; ret
    expect(run(hex("b8 ff ff ff ff 85 c0 78 07 48 c7 c0 02 00 00 00 c3"))).toBe(0xffffffffn);
  });

  it("reads jl after test as js, which is what test a,a defines it to be", () => {
    // mov $-1,%eax ; test %eax,%eax ; jl +7 ; mov $2,%rax ; ret
    expect(run(hex("b8 ff ff ff ff 85 c0 7c 07 48 c7 c0 02 00 00 00 c3"))).toBe(0xffffffffn);
  });
});

describe("the bit scans", () => {
  /*
   * `bsf` and `bsr` write the index of the lowest and highest set bit of the source. WASM has neither, and the two counts it does
   * have answer 64 for a zero source, where the processor leaves the destination exactly as it was, so the destination is picked
   * between its own value and the index.
   */
  it("finds the lowest set bit, which is 2 for 0x2c", () => {
    // mov $0x2c,%eax ; bsf %eax,%ecx ; mov %ecx,%eax ; ret
    expect(run(hex("b8 2c 00 00 00 0f bc c8 89 c8 c3"))).toBe(2n);
  });

  it("finds the highest set bit, which is 5 for the same 0x2c", () => {
    // mov $0x2c,%eax ; bsr %eax,%ecx ; mov %ecx,%eax ; ret
    expect(run(hex("b8 2c 00 00 00 0f bd c8 89 c8 c3"))).toBe(5n);
  });

  it("leaves the destination alone when the source is zero, and sets the zero flag", () => {
    /*
     * mov $0x2c,%ecx ; xor %eax,%eax ; bsr %eax,%ecx ; je found ; mov $99,%eax ; found: mov %ecx,%eax ; ret
     * the answer is the destination the scan left untouched, not the 63 a bare `clz` subtraction would have written
     */
    expect(run(hex("b9 2c 00 00 00 31 c0 0f bd c8 74 05 b8 63 00 00 00 89 c8 c3"))).toBe(0x2cn);
  });

  it("scans a 64-bit source at 64 bits, where the highest bit is 63", () => {
    // mov $1,%rax ; shl $63,%rax ; bsr %rax,%rcx ; mov %rcx,%rax ; ret
    expect(run(hex("48 c7 c0 01 00 00 00 48 c1 e0 3f 48 0f bd c8 48 89 c8 c3"))).toBe(63n);
  });

  it("refuses a condition the scan does not define, because it says nothing about the value", () => {
    // bsr %eax,%ecx ; js ...: the sign flag is undefined after a scan, which is not the sign of the index it wrote
    expect(() => run(hex("b8 2c 00 00 00 0f bd c8 78 05 b8 01 00 00 00 c3"))).toThrow(
      /bsf and bsr set the zero flag/,
    );
  });

  it("refuses the prefixed tzcnt by name rather than reading it as bsf", () => {
    // f3 0f bc c0 is `tzcnt %eax,%eax`, which answers the operand width for a zero source where bsf answers nothing
    expect(() => run(hex("31 c0 f3 0f bc c0 c3"))).toThrow(/tzcnt/);
  });
});

describe("what the flags are after an instruction", () => {
  it("answers a condition a comparison set after a shift, which no longer describes the flags", () => {
    /*
     * mov $0xff,%eax ; shr $4,%eax ; mov $3,%edx ; cmp $3,%edx ; ja skip ; mov $1,%eax ; skip: ret
     * the `ja` reads the comparison, not the shift: 3 is not above 3, so the branch is not taken and eax is 1
     */
    expect(run(hex("b8 ff 00 00 00 c1 e8 04 ba 03 00 00 00 83 fa 03 77 05 b8 01 00 00 00 c3"))).toBe(1n);
  });

  it("answers a second condition on the flags a branch kept, when the block is reached by falling out of the first", () => {
    /*
     * A graph body, because the `ja` is the first instruction of a block:
     *   mov $0,%eax ; mov $7,%edx ; cmp $5,%edx ; je end ; ja end ; sub $1,%edx ; jmp back to the cmp ; nop ; end: mov %edx,%eax ; ret
     * the `je` is not taken and the `ja` is, so eax is 7. Without carrying the flags along the fall-through the `ja` has no comparison
     * behind it, because a block start used to mean "the flags are unknown".
     */
    expect(run(hex("b8 00 00 00 00 ba 07 00 00 00 83 fa 05 74 08 77 06 83 ea 01 eb f4 90 89 d0 c3"))).toBe(7n);
  });
});

describe("a comparison at sixteen bits", () => {
  it("reads the sign from bit fifteen and not from bit sixty-three", () => {
    // mov $0x8000,%eax ; cmpw $0,%ax ; jl skip ; mov $7,%eax ; skip: ret  ->  the low half is negative, so the jump is taken
    expect(run(hex("b8 00 80 00 00 66 83 f8 00 7c 05 b8 07 00 00 00 c3"))).toBe(0x8000n);
  });

  it("compares as unsigned values, where the sign-extended reading would be wrong", () => {
    // mov $0x8000,%eax ; cmpw $1,%ax ; ja skip ; mov $7,%eax ; skip: ret
    // 0x8000 is above 1 as a word, and below it as a sign-extended value: keeping the operands masked is what makes this right
    expect(run(hex("b8 00 80 00 00 66 83 f8 01 77 05 b8 07 00 00 00 c3"))).toBe(0x8000n);
  });

  it("reads the sign of a masked test", () => {
    // mov $0x8000,%eax ; testw %ax,%ax ; js skip ; mov $7,%eax ; skip: ret
    expect(run(hex("b8 00 80 00 00 66 85 c0 78 05 b8 07 00 00 00 c3"))).toBe(0x8000n);
  });

  it("reads the sign of a negated word, which is the result and not the register above it", () => {
    // mov $5,%eax ; negw %ax ; js skip ; mov $7,%eax ; skip: ret  ->  0 - 5 is 0xfffb in the half, so the sign flag is set
    expect(run(hex("b8 05 00 00 00 66 f7 d8 78 05 b8 07 00 00 00 c3"))).toBe(0xfffbn);
  });
});

describe("the exchange", () => {
  it("swaps two registers, each getting the other's old value", () => {
    // mov $1,%eax ; mov $2,%ecx ; xchg %ecx,%eax ; sub %ecx,%eax ; ret -> 2 - 1, which is 1 and not -1
    expect(run(hex("b8 01 00 00 00 b9 02 00 00 00 87 c8 29 c8 c3"))).toBe(1n);
  });

  it("swaps the accumulator with the register the opcode names", () => {
    // mov $1,%eax ; mov $2,%ecx ; xchg %eax,%ecx (0x91) ; mov %ecx,%eax ; ret -> the accumulator's old value
    expect(run(hex("b8 01 00 00 00 b9 02 00 00 00 91 89 c8 c3"))).toBe(1n);
  });

  it("leaves the flags alone, so a branch before it still answers", () => {
    // mov $1,%eax ; cmp $1,%eax ; xchg %ecx,%eax ; jne skip ; mov $42,%eax ; ret
    // the comparison is still the last thing that wrote the flags, so the jump is not taken and the answer is 42
    expect(run(hex("b8 01 00 00 00 83 f8 01 87 c8 75 05 b8 2a 00 00 00 c3"))).toBe(0x2an);
  });
});

describe("the two increments", () => {
  /*
   * `inc` and `dec` are `add $1` and `sub $1` at every width, including a byte and a word in memory. The one thing they do not share
   * with an add is the carry, which the processor leaves alone: that is stated in the refusals below, since the operands a carry is
   * read from are replaced by the increment's own result.
   */
  it("increments and decrements a register", () => {
    // mov $5,%rax ; inc %rax ; ret
    expect(run(hex("48 c7 c0 05 00 00 00 48 ff c0 c3"))).toBe(6n);
    // mov $5,%ecx ; dec %ecx ; mov %ecx,%eax ; ret, which also checks the 32-bit write zeroes the upper half
    expect(run(hex("b9 05 00 00 00 ff c9 89 c8 c3"))).toBe(4n);
  });

  it("increments a byte and decrements one past zero", () => {
    // mov $5,%al ; inc %al ; ret
    expect(run(hex("b0 05 fe c0 c3"))).toBe(6n);
    // mov $0,%al ; dec %al ; ret: a byte decremented below zero wraps to all ones
    expect(run(hex("b0 00 fe c8 c3"))).toBe(0xffn);
  });

  it("increments a word without touching the rest of the register", () => {
    // mov $7,%ax ; inc %ax ; ret: the write is sixteen bits wide, and nothing above it is disturbed
    expect(run(hex("66 b8 07 00 66 ff c0 c3"))).toBe(8n);
  });

  it("sets the flags a branch reads, so an increment that reaches zero is a jump", () => {
    // mov $-1,%rax ; inc %rax ; je skip ; inc %rax ; ret: the first increment reaches zero and the branch is taken over the second
    expect(run(hex("48 c7 c0 ff ff ff ff 48 ff c0 74 03 48 ff c0 c3"))).toBe(0n);
  });

  it("refuses an adc after an increment, whose carry the processor leaves alone and this translator replaces", () => {
    // mov $1,%al ; cmp $2,%al ; inc %al ; adc $0,%rcx ; ret
    expect(() => run(hex("b0 01 80 f8 02 fe c0 48 83 d1 00 c3"))).toThrow(/adc reads the carry/);
  });
});

describe("a dword compare on sixteen bytes", () => {
  /*
   * `pcmpeqd` compares four dword lanes and makes each all ones or zero, which is how a compiler turns sixteen bytes into a mask. A
   * vector register here is two i64 halves, so the comparison is two halves, each an exclusive or: a lane is equal exactly where that
   * lane of the xor is zero. `movq` carries a half out so the test can see it.
   */
  const compare = (left: string, right: string): Uint8Array =>
    hex(`48 b8 ${left} 48 b9 ${right} 66 48 0f 6e c0 66 48 0f 6e c9 66 0f 76 c1 66 48 0f 7e c0 c3`);

  it("makes a lane all ones when the two dwords are equal", () => {
    // the four zero bytes are the high dword of each half: 5 against 5 in both lanes, so both are all ones and the half reads -1
    expect(run(compare("05 00 00 00 05 00 00 00", "05 00 00 00 05 00 00 00"))).toBe(-1n);
  });

  it("makes a lane zero when they differ", () => {
    // low lanes 5 and 6 differ, high lanes 5 and 5 are equal: the low dword is zero and the high one is all ones
    expect(run(compare("05 00 00 00 05 00 00 00", "06 00 00 00 05 00 00 00"))).toBe(-4294967296n);
  });

  it("answers each lane on its own, so one difference does not clear the other", () => {
    // low lanes both 5, high lanes 1 and 2: the equal low lane stays all ones while the other lane goes to zero
    expect(run(compare("05 00 00 00 01 00 00 00", "05 00 00 00 02 00 00 00"))).toBe(0xffffffffn);
  });

  it("makes a lane all ones when its low dword is all ones on both sides", () => {
    // 0xffffffff against 0xffffffff: a mask computed from the sign rather than from equality would answer zero here
    expect(run(compare("ff ff ff ff 00 00 00 00", "ff ff ff ff 00 00 00 00"))).toBe(-1n);
  });
});

describe("the two instructions that carry between steps", () => {
  /*
   * `adc` and `sbb` add or subtract the carry as well as their operand, which is how a compiler counts the things it has just
   * compared: `cmp $2,%al; adc $0,%rcx` adds one to a counter when the byte was below two. The carry is not kept as a flag but read
   * out of the comparison the flags still hold, which is the same borrow `sbb` uses and the same one a `jb` would read.
   */
  it("adds the carry when the comparison borrowed", () => {
    // mov $1,%al ; mov $5,%ecx ; cmp $2,%al ; adc $0,%rcx ; mov %rcx,%rax ; ret: 1 is below 2, so the counter gains one
    expect(run(hex("b0 01 b9 05 00 00 00 3c 02 48 83 d1 00 48 89 c8 c3"))).toBe(6n);
  });

  it("adds nothing when the comparison did not borrow", () => {
    // the same program comparing 3 against 2, which is not below it
    expect(run(hex("b0 03 b9 05 00 00 00 3c 02 48 83 d1 00 48 89 c8 c3"))).toBe(5n);
  });

  it("adds the carry and its immediate both", () => {
    // mov $1,%al ; mov $5,%ecx ; cmp $2,%al ; adc $0x10,%rcx ; mov %rcx,%rax ; ret
    expect(run(hex("b0 01 b9 05 00 00 00 3c 02 48 83 d1 10 48 89 c8 c3"))).toBe(0x16n);
  });

  it("adds no carry after a test, which subtracts nothing", () => {
    // mov $1,%al ; mov $5,%ecx ; test %al,%al ; adc $0,%rcx ; mov %rcx,%rax ; ret
    expect(run(hex("b0 01 b9 05 00 00 00 84 c0 48 83 d1 00 48 89 c8 c3"))).toBe(5n);
  });

  it("carries at the width it writes, so a byte adc stays inside the byte", () => {
    // mov $1,%al ; cmp $2,%al ; adc $0,%al ; ret: one plus the carry is two, at eight bits
    expect(run(hex("b0 01 3c 02 80 d0 00 c3"))).toBe(2n);
  });

  it("counts the characters of a string that fall in a range, which is the loop it was written for", () => {
    /*
     * The shape glibc's `strspn` uses: subtract the range's start, compare against its width, add the carry.
     * mov $1,%ecx ; the three bytes "0:;" counted one at a time ; mov %rcx,%rax ; ret
     */
    const count = (byte: number): bigint => {
      const body =
        `b9 00 00 00 00 b0 ${byte.toString(16)} 80 e8 30 80 f8 0a 48 83 d1 00 48 89 c8 c3`;
      return run(hex(body));
    };
    expect(count(0x30)).toBe(1n);
    expect(count(0x3a)).toBe(0n);
  });
});

describe("a shift of a byte", () => {
  /*
   * The byte encodings of the shift group, `0xc0`, `0xd0` and `0xd2`, which a walk of glibc needed and which differ from the wider
   * forms in the count rather than in the shift: the processor masks a count to five bits whatever the operand's width, so `shr $9,%al`
   * shifts by nine and leaves zero.
   */
  it("shifts a byte right by an immediate", () => {
    // mov $0xff,%al ; shr $4,%al ; ret
    expect(run(hex("b0 ff c0 e8 04 c3"))).toBe(0x0fn);
  });

  it("wraps a byte shift at eight bits", () => {
    // mov $0x80,%al ; shl $1,%al ; jne skip ; mov $42,%eax ; ret: 0x80 doubled is 0x100, which a byte holds as zero
    expect(run(hex("b0 80 c0 e0 01 75 05 b8 2a 00 00 00 c3"))).toBe(0x2an);
  });

  it("keeps the sign through an arithmetic shift of a byte", () => {
    // mov $0x80,%al ; sar $1,%al ; ret: shifting a negative byte right gives 0xc0, and a logical shift would give 0x40
    expect(run(hex("b0 80 c0 f8 01 c3"))).toBe(0xc0n);
  });

  it("masks the count the way the processor does, to five bits and not to the operand's width", () => {
    // mov $0xff,%al ; shr $9,%al ; ret: nine bits out of eight leaves nothing, where a count masked to three bits would leave 0x7f
    expect(run(hex("b0 ff c0 e8 09 c3"))).toBe(0x00n);
  });

  it("takes its count from %cl and its flags from the byte it wrote", () => {
    // mov $0xff,%al ; mov $8,%cl ; shl %cl,%al ; jne skip ; mov $42,%eax ; ret: eight bits out of a byte is zero
    expect(run(hex("b0 ff b1 08 d2 e0 75 05 b8 2a 00 00 00 c3"))).toBe(0x2an);
  });

  it("shifts a byte by one through the short encoding", () => {
    // mov $0x01,%al ; shl $1,%al (0xd0) ; ret
    expect(run(hex("b0 01 d0 e0 c3"))).toBe(0x02n);
  });

  it("answers a sign condition from the byte it shifted", () => {
    // mov $0x80,%al ; sar $1,%al ; js skip ; mov $42,%eax ; ret: 0xc0 is negative at eight bits, so the jump is taken
    expect(run(hex("b0 80 c0 f8 01 78 05 b8 2a 00 00 00 c3"))).toBe(0xc0n);
  });
});

describe("the sign of a byte", () => {
  /*
   * A byte is sign-extended for the sake of its flags, and the extension has to reach bit 63 of the lane it is kept in: a shift by
   * 24 puts bit 7 at bit 31 instead, which is inside a 64-bit lane, so the arithmetic shift that follows leaves the byte positive and
   * `js` never fires. Every byte whose sign mattered was answering this way until a byte shift had to read one.
   */
  it("takes js after negating a byte that came out negative", () => {
    // mov $0x80,%al ; neg %al ; js skip ; mov $42,%eax ; ret: negating -128 leaves 0x80, whose sign bit is set
    expect(run(hex("b0 80 f6 d8 78 05 b8 2a 00 00 00 c3"))).toBe(0x80n);
  });

  it("takes js after an and leaves a byte negative", () => {
    // mov $0xff,%al ; and $0x80,%al ; js skip ; mov $42,%eax ; ret: 0xff and 0x80 is 0x80, which is negative at eight bits
    expect(run(hex("b0 ff 24 80 78 05 b8 2a 00 00 00 c3"))).toBe(0x80n);
  });
});

describe("the compare and swap", () => {
  /*
   * The shape a one-time guard is written in: compare, and write only if nothing has claimed it yet. The register form is used
   * here because it needs no memory to be set up, and both arms are checked because they are the two answers it exists to give.
   */
  it("writes the source into the destination when the accumulator matches it", () => {
    // mov $5,%eax ; mov $5,%ecx ; mov $42,%edx ; cmpxchg %edx,%ecx ; jne +2 ; mov %ecx,%eax ; ret
    expect(run(hex("b8 05 00 00 00 b9 05 00 00 00 ba 2a 00 00 00 0f b1 d1 75 02 89 c8 c3"))).toBe(0x2an);
  });

  it("loads the destination into the accumulator when they differ, and leaves the destination alone", () => {
    // the same bytes with a different accumulator, and `je` where the `jne` was: on a mismatch the move is the one that runs, so
    // the answer is the destination's old value. If the source had been written where the destination was, it would be 42.
    expect(run(hex("b8 07 00 00 00 b9 05 00 00 00 ba 2a 00 00 00 0f b1 d1 74 02 89 c8 c3"))).toBe(5n);
  });

  it("leaves the flags describing the comparison, so a branch on order answers too", () => {
    // mov $9,%eax ; mov $5,%ecx ; mov $42,%edx ; cmpxchg %edx,%ecx ; ja skip ; mov $1,%eax ; ret
    // 9 is above 5, so the jump is taken and the accumulator keeps the destination's value: a translator that read the flags the
    // other way round would answer 1
    expect(run(hex("b8 09 00 00 00 b9 05 00 00 00 ba 2a 00 00 00 0f b1 d1 77 05 b8 01 00 00 00 c3"))).toBe(5n);
  });
});

describe("the widenings a compiler reads a field with", () => {
  it("zero-extends a byte register, which keeps the byte and drops the rest", () => {
    // mov $0x2c,%eax ; movzbl %al,%ecx ; mov %ecx,%eax ; ret
    expect(run(hex("b8 2c 00 00 00 0f b6 c8 89 c8 c3"))).toBe(0x2cn);
  });

  it("sign-extends a byte register, so 0x80 becomes -128 rather than 128", () => {
    // mov $0x80,%eax ; movsbl %al,%ecx ; mov %ecx,%eax ; ret  ->  a 32-bit write zero-extends, so the answer is 0xffffff80
    expect(run(hex("b8 80 00 00 00 0f be c8 89 c8 c3"))).toBe(0xffffff80n);
  });
});

describe("a condition written as a byte", () => {
  it("writes one when the condition holds", () => {
    // mov $5,%eax ; cmp $5,%eax ; sete %cl ; movzbl %cl,%ecx ; mov %ecx,%eax ; ret
    expect(run(hex("b8 05 00 00 00 83 f8 05 0f 94 c1 0f b6 c9 89 c8 c3"))).toBe(1n);
  });

  it("writes zero when it does not", () => {
    // the same with 6 against 5: `sete` is false, so the answer is the byte it wrote
    expect(run(hex("b8 06 00 00 00 83 f8 06 83 f8 05 0f 94 c1 0f b6 c9 89 c8 c3"))).toBe(0n);
  });
});

describe("branches nest as structure", () => {
  it("nests one conditional branch inside another and branches out of both", () => {
    /*
     * mov $1,%eax ; cmp $1,%eax ; jne end ; mov $2,%eax ; cmp $2,%eax ; je end ; mov $3,%eax ; end: ret
     * the outer branch wraps the inner one, and the inner je leaves both: rax is 2
     */
    expect(run(hex("b8 01 00 00 00 83 f8 01 75 0f b8 02 00 00 00 83 f8 02 74 05 b8 03 00 00 00 c3"))).toBe(2n);
  });

  it("runs the code under the inner branch when its condition fails", () => {
    // the same program with `cmp $3,%eax`, so the inner je is not taken and rax ends at 3
    expect(run(hex("b8 01 00 00 00 83 f8 01 75 0f b8 02 00 00 00 83 f8 03 74 05 b8 03 00 00 00 c3"))).toBe(3n);
  });

  it("translates the if/else shape a compiler emits, where the then-branch ends in a forward jmp", () => {
    /*
     * mov $1,%eax ; cmp $1,%eax ; jne else ; mov $10,%eax ; jmp end ; else: mov $20,%eax ; end: ret
     * the equal case takes the then-branch and jumps over the else
     */
    expect(run(hex("b8 01 00 00 00 83 f8 01 75 0c b8 0a 00 00 00 eb 05 b8 14 00 00 00 c3"))).toBe(10n);
  });

  it("takes the else arm when the condition fails", () => {
    // the same program comparing against 2, so the jne jumps to the else arm at offset 17 and never returns
    expect(run(hex("b8 01 00 00 00 83 f8 02 75 07 b8 0a 00 00 00 eb 05 b8 14 00 00 00 c3"))).toBe(20n);
  });
});
