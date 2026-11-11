# The byte registers: `%dh` is bits 8 to 15 of `%rdx`, and writing it must leave the bytes around it alone.
.globl target
target:
    mov $0x1234, %edx
    test $0x02, %dh           # 0x12 has bit 1 set, so the zero flag is clear
    je wrong
    test $0x40, %dh           # and bit 6 clear
    jne wrong

    mov $0xff, %dh            # writing the high byte leaves the low byte where it was
    cmp $0x12ff, %edx
    jne wrong
    mov $0x00, %dl            # and writing the low byte leaves the high one
    cmp $0xff00, %edx
    jne wrong

    mov $0xabcd, %eax         # %ah is 0xab, %al is 0xcd, and neither is a register of its own
    mov %ah, %bl
    cmp $0xab, %bl
    jne wrong
    mov $0x00, %ah
    cmp $0x00cd, %eax
    jne wrong

    mov $0x80, %dh            # -128 in a byte
    mov $0x01, %dl
    cmp %dh, %dl              # 1 is greater than -128 signed, and greater unsigned too
    jle wrong
    ja wrong
    mov $0x7f, %dh            # 127, still below 129 unsigned
    mov $0x81, %dl
    cmp %dh, %dl              # 129 - 127 borrows nothing, but as bytes 129 is -127
    jle wrong

    mov $42, %rax
    ret
wrong:
    mov $1, %rax
    ret
