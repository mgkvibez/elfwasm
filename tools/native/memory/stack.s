# The stack, used the way a compiled function uses it: a frame pointer, a local slot, and a push/pop pair.
#
# The translator sets %rsp above the image and checks a floor, so the numbers only agree with the processor if pushes and
# pops move the pointer by the right amount in the right direction, and if a spill to -8(%rbp) lands in the frame.
.globl target
target:
    push %rbp
    mov %rsp, %rbp
    sub $0x10, %rsp
    movq $9, -8(%rbp)     # a local variable in the frame
    mov -8(%rbp), %rax    # read it back
    push %rax             # and through the stack proper
    mov $1, %rax
    pop %rdx
    add %rdx, %rax        # 1 + 9
    leave                 # %rsp = %rbp, then pop %rbp: the frame is gone
    ret
.size target, .-target
