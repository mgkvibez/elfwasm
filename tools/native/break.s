# Sums 1 + 2 + 3 + 4 and leaves as soon as the sum reaches 9, so 10.
.globl target
target:
    xor %eax, %eax
    xor %ecx, %ecx
loop:
    add $1, %ecx
    add %ecx, %eax
    cmp $9, %eax
    jge done
    cmp $10, %ecx
    jl loop
done:
    ret
