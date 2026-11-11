# Conditional moves, which is what a glibc ifunc resolver reaches once it has compared: the destination keeps its value unless
# the condition holds, and the flags survive, so the same comparison can drive the branch that follows.
.globl target
target:
    # below: 2 is less than 3, so %rax stays and %rbx takes the source
    mov $0x11, %rax
    mov $0x22, %rbx
    cmp $3, %rax
    cmovl %rbx, %rax          # 0x11 < 3 is false, so %rax keeps 0x11
    cmp $0x11, %rax
    jne wrong
    mov $2, %rax
    mov $0x22, %rbx
    cmp $3, %rax
    cmovl %rbx, %rax          # taken: %rax becomes 0x22
    cmp $0x22, %rax
    jne wrong

    # the flags survive a cmov, so a branch after it still sees the comparison
    mov $5, %rax
    mov $9, %rbx
    cmp $5, %rax
    cmove %rbx, %rax          # equal, so %rax becomes 9
    jne wrong
    cmp $9, %rax
    jne wrong

    # the unsigned pair: 0xffffffffffffffff is the largest value unsigned and -1 signed
    mov $0xffffffffffffffff, %rax
    mov $1, %rbx
    cmp $0, %rax
    cmova %rbx, %rax          # greater unsigned, so %rax becomes 1
    cmp $1, %rax
    jne wrong
    mov $0xffffffffffffffff, %rax
    mov $1, %rbx
    cmp $0, %rax
    cmovl %rbx, %rax          # less than signed (-1 < 0), so %rax becomes 1 as well
    cmp $1, %rax
    jne wrong
    mov $0xffffffffffffffff, %rax
    mov $1, %rbx
    cmp $0, %rax
    cmovg %rbx, %rax          # greater signed is false, so nothing is written
    mov $-1, %rcx
    cmp %rcx, %rax
    jne wrong

    # 32 bits: the write clears the upper half, so the answer is the source and not what used to be above it
    mov $0x1111111111111111, %rax
    mov $0xffffff00, %ecx
    cmp %ecx, %eax            # 0x11 > 7 unsigned, so a `b` is not taken
    cmovb %ebx, %eax
    mov $0x1111111111111111, %rcx
    cmp %rcx, %rax
    jne wrong
    mov $0x1111111111111111, %rax
    mov $0x22222222, %ebx
    mov $0x00ffffff, %ecx
    cmp %ecx, %eax            # 0x11 < 0xffffff, so unsigned smaller holds
    cmovb %ebx, %eax
    cmp $0x22222222, %rax     # and the upper half is cleared rather than kept
    jne wrong

    # the equal case again, through the 32-bit form, where the source register is the whole of what is written
    mov $0x00, %rax
    mov $0x77, %ebx
    cmp $0, %rax
    cmove %ebx, %eax          # taken, and the upper half of %rax goes with it
    cmp $0x77, %rax
    jne wrong

    # the flags survive a branch: a comparison made before an if/else is still there after the join, which is what a resolver
    # does when it compares an address and picks a routine on each side
    mov $0, %rcx
    mov $1, %rax
    mov $0xaa, %rbx
    cmp $2, %rax              # not equal, so the zero flag is clear going into the branch
    je other
    mov $0x11, %rdx
    jmp after
other:
    mov $0x22, %rdx
after:
    cmovne %rbx, %rcx         # the comparison from before the branch, still readable: %rcx becomes 0xaa
    cmp $0xaa, %rcx
    jne wrong

    mov $0, %rcx
    mov $0x55, %rbx
    mov $4, %rax
    cmp $4, %rax
    cmove %rbx, %rcx          # equal, so %rcx becomes 0x55
    cmp $0x55, %rcx
    jne wrong

    mov $7, %rax
    ret
wrong:
    mov $1, %rax
    ret
