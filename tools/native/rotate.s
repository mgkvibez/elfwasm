# Does a rotate put each bit where the architecture says, and leave the flags a comparison left alone?
#
# Every case compares against a constant that can be read off the definition: rotating right by four moves the low nibble to the top, and
# a rotate by the width's own count is no rotation at all. 42 is every case agreeing; any other number is the number of the case that
# did not, so a failure names itself rather than needing a second run.
    .macro expect value, case
    movabs $\value, %rcx
    cmp %rcx, %rax
    jne .Lfail
    .endm

    .globl target
target:
    # a quadword, rotated right by four: the low nibble goes to the top
    movabs $0x1234, %rax
    ror $4, %rax
    mov $1, %r8d
expect 0x4000000000000123

    # the same value rotated left by four, which is the same thing the other way
    movabs $0x1234, %rax
    rol $4, %rax
    mov $2, %r8d
expect 0x12340

    # a byte rotates inside its own eight bits, not inside the register's 64
    mov $0x11, %al
    ror $1, %al
    mov $3, %r8d
expect 0x88

    # and a count equal to the width is no rotation: the count is reduced to the width before it is used
    mov $0x11, %al
    ror $8, %al
    mov $4, %r8d
expect 0x11

    # the swap-halves idiom, which is what a compiler emits for a byte-order swap of two dwords
    movabs $0x0123456789abcdef, %rax
    ror $32, %rax
    mov $5, %r8d
expect 0x89abcdef01234567

    # a word, rotated through the count in %cl rather than by an immediate
    movabs $0x0000000000001234, %rax
    mov $4, %cl
    ror %cl, %ax
    mov $6, %r8d
expect 0x4123

    # the point of the flags: a rotate does not touch the zero, sign or parity flags, so a branch after it still answers the
    # comparison that came before
    mov $7, %r8d
    mov $5, %eax
    cmp $5, %eax
    ror $3, %rax
    je .Lflags_kept
    mov $7, %eax
    ret

.Lflags_kept:
    # and the same for a sign flag set by a comparison, read after a rotate left
    mov $8, %r8d
    mov $-8, %eax
    cmp $0, %eax
    rol $2, %eax
    js .Ldone
    mov $8, %eax
    ret

.Ldone:
    mov $42, %eax
    ret

.Lfail:
    mov %r8d, %eax
    ret
