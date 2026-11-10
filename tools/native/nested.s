# An inner loop that adds 2, inside an outer loop that runs 3 times, so 6.
.globl target
target:
    xor %eax, %eax
    xor %ecx, %ecx
outer:
    xor %edx, %edx
inner:
    add $1, %edx
    add $1, %eax
    cmp $2, %edx
    jl inner
    add $1, %ecx
    cmp $3, %ecx
    jl outer
    ret
