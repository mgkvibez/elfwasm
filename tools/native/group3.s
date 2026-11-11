# Group 3 at three widths: test against an immediate, not, and neg. The byte forms are what a glibc ifunc resolver opens with,
# `test $0x48, %dh`, and the wider ones are what the same resolver reaches for once it has decided.
.globl target
target:
    mov $0x1234, %edx
    test $0x02, %dl           # bit 1 of 0x34 is set, so the zero flag is clear
    je wrong
    test $0x04, %dl           # bit 2 of 0x34 is clear
    jne wrong
    mov $0x00, %cl
    test $0x00, %cl           # a mask of zero always sets the zero flag
    jne wrong
    mov $0x00, %cl
    test $0x01, %cl           # and a mask that misses the value leaves it set
    jne wrong

    mov $0x00, %al
    not %al                   # 8-bit not is 0xff, and the rest of %rax stays zero
    cmp $0xff, %al
    jne wrong
    mov $0x80, %al
    neg %al                   # -128 negated is -128, which is its own negation in eight bits
    cmp $0x80, %al
    jne wrong
    mov $0x01, %al
    neg %al                   # and the zero flag comes with the result: -1 is not zero
    je wrong
    mov $0x00, %al
    neg %al
    jne wrong

    mov $0x0f, %eax
    not %eax                  # 32-bit not fills the low 32 bits and leaves the high half zero
    mov $0xfffffff0, %ebx
    cmp %rbx, %rax
    jne wrong
    mov $1, %eax
    neg %eax
    mov $0xffffffff, %ebx
    cmp %rbx, %rax
    jne wrong

    mov $0x00000000ffffffff, %rax
    not %rax                  # 64-bit not is the whole register
    mov $0xffffffff00000000, %rbx
    cmp %rbx, %rax
    jne wrong
    mov $1, %rax
    neg %rax
    cmp $0xffffffffffffffff, %rax
    jne wrong

    mov $7, %rax
    ret
wrong:
    mov $1, %rax
    ret
