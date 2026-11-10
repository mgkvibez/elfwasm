# The compiler's if/else shape: a conditional jump to the else arm, and a jmp over it at the end.
.globl target
target:
    mov $1, %eax
    cmp $1, %eax
    jne else
    mov $10, %ecx
    jmp done
else:
    mov $20, %ecx
done:
    mov %ecx, %eax
    ret
