# Five passes of a decrease-and-test loop; the count is the result, so 5.
.globl target
target:
    mov $5, %eax
    xor %ecx, %ecx
loop:
    add $1, %ecx
    sub $1, %eax
    cmp $0, %eax
    jne loop
    mov %ecx, %eax
    ret
