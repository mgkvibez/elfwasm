# The same bytes with jb: unsigned 0xffffffff is not below 1, so it is not taken and 2 is returned.
.globl target
target:
    mov $-1, %eax
    cmp $1, %eax
    jb below
    mov $2, %ecx
    jmp done
below:
    mov $3, %ecx
done:
    mov %ecx, %eax
    ret
