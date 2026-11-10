# 32-bit signed: -1 is below 1, so the jl is taken and 3 is returned.
.globl target
target:
    mov $-1, %eax
    cmp $1, %eax
    jl below
    mov $2, %ecx
    jmp done
below:
    mov $3, %ecx
done:
    mov %ecx, %eax
    ret
