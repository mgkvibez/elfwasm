# The subset a musl hello world uses, in the order it uses it. Assembled by tools/build-decode-fixture.sh, and the
# listing objdump produces next to it is the expectation the decoder is tested against: not a hand-written guess.
    .text
    .globl _start
_start:
    movq $1, %rax              # write
    movq $1, %rdi              # stdout
    leaq message(%rip), %rsi   # the string, addressed rip-relative
    movq $13, %rdx             # its length
    syscall
    xorl %eax, %eax            # exit code zero
    pushq %rbp                 # a prologue, so the stack frame is real
    movq %rsp, %rbp
    subq $16, %rsp
    movq $0, -8(%rbp)          # a local through a displacement
    movl $5, %eax              # a 32-bit move, zero-extending
    movl %eax, (%rsp)          # store through a SIB byte with no base displacement
    movq 8(%rsp,%rax,4), %rcx  # load with base, index and scale
    addq $2, %rax
    addq $1000, %rbx
    subq $3, %rcx
    cmpq $10, %rax
    jne .skip
    call bump
.skip:
    je .done
    jmp .done
.done:
    nop
    popq %rbp
    ret
bump:
    addq $1, %rax
    ret
    # the second slice: what a compiler emits once the prologue is done
    endbr64
    testq %rax, %rax
    testl $255, %eax
    testq $4096, %rcx
    movzbl %al, %ecx
    movzbl 3(%rdi), %edx
    movzwl 4(%rdi), %edx
    movsbq %al, %rdx
    movslq %eax, %r8
    movsxd %r8d, %r9
    imulq %rcx, %rax
    imulq $7, %r8, %r9
    imull $3, %eax, %edx
    negq %rax
    notq %rcx
    setne %al
    setb %cl
    sete %r10b
    shlq $3, %rax
    shrl $1, %eax
    sarq %cl, %rdx
    andq %rcx, %rax
    orq %rdx, %rax
    andl $15, %eax
    orq $64, %rax
    xorq $255, %rax
    addq %rcx, %rax
    subq %rcx, %r8
    cmpq %rcx, %rax
    cmpl $5, %eax
    pushq $42
    pushq $1000
    cwde
    cltq
    cdq
    cqto
    leave
    int3
    ud2
    jo .done
    jns .skip
    jg .done
    .section .rodata
message:
    .ascii "hello, world\n"
