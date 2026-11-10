# A value that has to go through memory, so the translator's stores and loads are checked by the processor too.
#
# Both accesses are absolute, which is the addressing mode a static non-PIE binary uses for a global: the address is
# resolved by the linker and sits in the instruction, which is exactly what makes it translatable at load time.
.globl target
target:
    movq $0x2a, slot      # store 42
    mov slot, %rax        # read it back
    ret
.size target, .-target

.data
.align 8
slot:
    .quad 0
.size slot, .-slot
