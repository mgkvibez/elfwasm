# Reads the flags left by arithmetic rather than by a comparison, which is what a compiler's canary check does:
# `sub %fs:0x28, %rdx` followed by `jne`. Three conditions, three answers, and the third one has to be the one returned.
.globl target
target:
    mov $5, %rax
    sub $7, %rax          # -2, so a signed comparison against zero is below
    jge wrong
    mov $1, %rax
    sub $2, %rax          # 1 - 2 borrows, so an unsigned comparison is below
    jae wrong
    mov $6, %rax
    add $6, %rax          # 12, which is neither equal to nor below zero
    jle wrong
    je wrong
    ret
wrong:
    mov $1, %rax
    ret
