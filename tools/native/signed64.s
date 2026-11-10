# 64-bit signed: -1 is below 1 here too, and the comparison is made at 64 bits.
.globl target
target:
    mov $-1, %rax
    cmp $1, %rax
    jl below
    mov $2, %ecx
    jmp done
below:
    mov $3, %ecx
done:
    mov %ecx, %eax
    ret
