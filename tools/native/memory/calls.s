# One function calling another, which is the calling convention in its smallest form.
#
# The argument goes in %rdi and the result comes back in %rax, both of which are module globals in the translated version.
# Nothing here touches memory, so the translated module needs no memory and no stack: if the numbers still agree, then the
# registers really are shared across the call and the return really does come back from the callee.
.globl target
target:
    mov $7, %rdi
    call helper
    add $1, %rax
    ret
.size target, .-target

helper:
    mov %rdi, %rax
    add $5, %rax
    ret
.size helper, .-helper
