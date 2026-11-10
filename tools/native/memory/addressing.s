# Addressing through a register, which is how a pointer is used in real code.
#
# The address of `slot` is loaded into a register, read through, added to, written back through the same register, and then
# a second read uses a displacement from that register. Every one of those is a run-time address the translator has to
# subtract the image base from and bound-check.
.globl target
target:
    mov $slot, %rcx       # rcx holds the virtual address of slot
    mov (%rcx), %rax      # read it through the register
    add $7, %rax
    mov %rax, (%rcx)      # write it back through the same register
    mov 8(%rcx), %rdx     # and read its neighbour by displacement
    add %rdx, %rax        # 7 + 5
    ret
.size target, .-target

.data
.align 8
slot:
    .quad 0
.size slot, .-slot
.align 8
neighbour:
    .quad 5
.size neighbour, .-neighbour
