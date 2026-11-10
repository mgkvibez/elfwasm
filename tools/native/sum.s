# 1 + 2 + 3 + 4, accumulated in eax while ecx counts, so 10.
.globl target
target:
    xor %eax, %eax
    xor %ecx, %ecx
loop:
    add $1, %ecx
    add %ecx, %eax
    cmp $4, %ecx
    jl loop
    ret
