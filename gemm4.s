	.file "gemm.c"
	.text
..TXTST0:
# -- Begin  sgemm_opt, L_sgemm_opt_25__par_region0_2.0
	.text
# mark_begin;
# Threads 2
        .align    16,0x90
# --- sgemm_opt(const uint64_t, const uint64_t, const uint64_t, const float *__restrict__, const float *__restrict__, float *__restrict__)
sgemm_opt:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %r8
# parameter 6: %r9
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_sgemm_opt.1:
..L2:
                                                          #14.1
        pushq     %rbp                                          #14.1
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp                                    #14.1
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp                                    #14.1
        subq      $896, %rsp                                    #14.1 c1
        movq      %rdi, (%rsp)                                  #14.1 c3
        movl      $.2.5_2_kmpc_loc_struct_pack.12, %edi         #25.1 c3
        movq      %rbx, 48(%rsp)                                #14.1[spill] c3
        movq      %r15, 16(%rsp)                                #14.1[spill] c5
        movq      %r14, 24(%rsp)                                #14.1[spill] c5
        movq      %r13, 32(%rsp)                                #14.1[spill] c7
        movq      %r12, 40(%rsp)                                #14.1[spill] c7
        movq      %rsi, 8(%rsp)                                 #14.1 c9
        movq      %rdx, 56(%rsp)                                #14.1 c9
        movq      %rcx, 64(%rsp)                                #14.1 c11
        movq      %r8, 72(%rsp)                                 #14.1 c11
        movq      %r9, 80(%rsp)                                 #14.1 c13
        call      __kmpc_global_thread_num                      #25.1 c13
	.cfi_escape 0x10, 0x03, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xb0, 0xfc, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xa8, 0xfc, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xa0, 0xfc, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x98, 0xfc, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0f, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x90, 0xfc, 0xff, 0xff, 0x22
                                # LOE eax
..B1.63:                        # Preds ..B1.1
                                # Execution count [1.00e+00]
        movl      %eax, 88(%rsp)                                #25.1 c1
        movl      $.2.5_2_kmpc_loc_struct_pack.20, %edi         #25.1 c1
        xorl      %eax, %eax                                    #25.1 c1
..___tag_value_sgemm_opt.11:
        call      __kmpc_ok_to_fork                             #25.1
..___tag_value_sgemm_opt.12:
                                # LOE eax
..B1.2:                         # Preds ..B1.63
                                # Execution count [1.00e+00]
        testl     %eax, %eax                                    #25.1 c1
        je        ..B1.5        # Prob 50%                      #25.1 c3
                                # LOE
..B1.3:                         # Preds ..B1.2
                                # Execution count [0.00e+00]
        movl      $.2.5_2_kmpc_loc_struct_pack.20, %edi         #25.1 c1
        movl      88(%rsp), %esi                                #25.1 c1
        movl      $64, %edx                                     #25.1 c1
        xorl      %eax, %eax                                    #25.1 c3
..___tag_value_sgemm_opt.13:
        call      __kmpc_push_num_threads                       #25.1
..___tag_value_sgemm_opt.14:
                                # LOE
..B1.4:                         # Preds ..B1.3
                                # Execution count [0.00e+00]
        addq      $-32, %rsp                                    #25.1 c1
        movl      $L_sgemm_opt_25__par_region0_2.0, %edx        #25.1 c1
        lea       32(%rsp), %rcx                                #25.1 c3
        movl      $.2.5_2_kmpc_loc_struct_pack.20, %edi         #25.1 c3
        movl      $6, %esi                                      #25.1 c3
        lea       56(%rcx), %rax                                #25.1 c5
        movq      %rax, (%rsp)                                  #25.1 c7
        lea       8(%rcx), %r8                                  #25.1 c7
        lea       80(%rcx), %r9                                 #25.1 c9
        lea       8(%rax), %rbx                                 #25.1 c9
        movq      %rbx, 8(%rsp)                                 #25.1 c11
        lea       16(%rax), %r10                                #25.1 c11
        movq      %r10, 16(%rsp)                                #25.1 c13
        xorl      %eax, %eax                                    #25.1 c13
..___tag_value_sgemm_opt.15:
        call      __kmpc_fork_call                              #25.1
..___tag_value_sgemm_opt.16:
                                # LOE
..B1.64:                        # Preds ..B1.4
                                # Execution count [0.00e+00]
        addq      $32, %rsp                                     #25.1 c1
        jmp       ..B1.8        # Prob 100%                     #25.1 c1
                                # LOE
..B1.5:                         # Preds ..B1.2
                                # Execution count [0.00e+00]
        movl      $.2.5_2_kmpc_loc_struct_pack.20, %edi         #25.1 c1
        movl      88(%rsp), %esi                                #25.1 c1
        xorl      %eax, %eax                                    #25.1 c1
..___tag_value_sgemm_opt.17:
        call      __kmpc_serialized_parallel                    #25.1
..___tag_value_sgemm_opt.18:
                                # LOE
..B1.6:                         # Preds ..B1.5
                                # Execution count [0.00e+00]
        addq      $-16, %rsp                                    #25.1 c1
        movl      $___kmpv_zerosgemm_opt_0, %esi                #25.1 c1
        lea       16(%rsp), %rdx                                #25.1 c3
        lea       104(%rsp), %rdi                               #25.1 c3
        lea       64(%rdx), %rax                                #25.1 c5
        movq      %rax, (%rsp)                                  #25.1 c7
        lea       8(%rdx), %rcx                                 #25.1 c7
        lea       80(%rdx), %r8                                 #25.1 c9
        lea       56(%rdx), %r9                                 #25.1 c9
        lea       8(%rax), %rbx                                 #25.1 c11
        movq      %rbx, 8(%rsp)                                 #25.1 c13
..___tag_value_sgemm_opt.19:
        call      L_sgemm_opt_25__par_region0_2.0               #25.1
..___tag_value_sgemm_opt.20:
                                # LOE
..B1.65:                        # Preds ..B1.6
                                # Execution count [0.00e+00]
        addq      $16, %rsp                                     #25.1 c1
                                # LOE
..B1.7:                         # Preds ..B1.65
                                # Execution count [0.00e+00]
        movl      $.2.5_2_kmpc_loc_struct_pack.20, %edi         #25.1 c1
        movl      88(%rsp), %esi                                #25.1 c1
        xorl      %eax, %eax                                    #25.1 c1
..___tag_value_sgemm_opt.21:
        call      __kmpc_end_serialized_parallel                #25.1
..___tag_value_sgemm_opt.22:
                                # LOE
..B1.8:                         # Preds ..B1.64 ..B1.7
                                # Execution count [1.00e+00]
        movq      16(%rsp), %r15                                #281.1[spill] c1
	.cfi_restore 15
        movq      24(%rsp), %r14                                #281.1[spill] c1
	.cfi_restore 14
        movq      32(%rsp), %r13                                #281.1[spill] c5 stall 1
	.cfi_restore 13
        movq      40(%rsp), %r12                                #281.1[spill] c5
	.cfi_restore 12
        movq      48(%rsp), %rbx                                #281.1[spill] c9 stall 1
	.cfi_restore 3
        movq      %rbp, %rsp                                    #281.1 c11
        popq      %rbp                                          #281.1
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret                                                     #281.1
	.cfi_def_cfa 6, 16
                                # LOE
L_sgemm_opt_25__par_region0_2.0:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %r8
# parameter 6: %r9
# parameter 7: 16 + %rbp
# parameter 8: 24 + %rbp
..B1.9:                         # Preds ..B1.0
                                # Execution count [0.00e+00]
        pushq     %rbp                                          #25.1
	.cfi_def_cfa 7, 16
        movq      %rsp, %rbp                                    #25.1
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp                                    #25.1
        subq      $896, %rsp                                    #25.1 c1
        movq      %rbx, 48(%rsp)                                #25.1[spill] c3
	.cfi_escape 0x10, 0x03, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xb0, 0xfc, 0xff, 0xff, 0x22
        movq      %rcx, %rbx                                    #25.1 c3
        movq      %r15, 16(%rsp)                                #25.1[spill] c3
	.cfi_escape 0x10, 0x0f, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x90, 0xfc, 0xff, 0xff, 0x22
        movq      24(%rbp), %r15                                #25.1 c5
        movq      %r14, 24(%rsp)                                #25.1[spill] c5
	.cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x98, 0xfc, 0xff, 0xff, 0x22
        movq      %r9, %r14                                     #25.1 c5
        movq      %r13, 32(%rsp)                                #25.1[spill] c7
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xa0, 0xfc, 0xff, 0xff, 0x22
        movq      %rdx, %r13                                    #25.1 c7
        movq      %r12, 40(%rsp)                                #25.1[spill] c9
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xa8, 0xfc, 0xff, 0xff, 0x22
        movq      16(%rbp), %r12                                #25.1 c9
        movq      %r8, (%rsp)                                   #25.1[spill] c11
#       omp_get_thread_num(void)
        call      omp_get_thread_num                            #25.1 c11
                                # LOE rbx r12 r13 r14 r15 eax
..B1.66:                        # Preds ..B1.9
                                # Execution count [0.00e+00]
        movq      (%rsp), %r8                                   #[spill] c1
                                # LOE rbx r8 r12 r13 r14 r15 eax
..B1.10:                        # Preds ..B1.66
                                # Execution count [1.00e+00]
        movq      (%r12), %r9                                   #25.1 c1
        movq      %r9, 432(%rsp)                                #25.1[spill] c5 stall 1
        movq      (%rbx), %r12                                  #25.1 c5
        movl      %eax, %r10d                                   #30.34 c5
        movl      %eax, %r11d                                   #31.33 c5
        movq      (%r13), %rcx                                  #25.1 c7
        sarl      $5, %r10d                                     #30.34 c7
        andl      $31, %r11d                                    #31.33 c7
        movq      (%r15), %rdi                                  #25.1 c9
        movq      %rdi, 392(%rsp)                               #25.1[spill] c13 stall 1
        movq      %r12, %rdx                                    #34.35 c13
        movq      %rcx, %rbx                                    #33.34 c13
        movq      (%r14), %r9                                   #25.1 c13
        shrq      $5, %rdx                                      #34.35 c15
        shrq      $1, %rbx                                      #33.34 c15
        movq      (%r8), %rdi                                   #25.1 c15
        addq      $1, %rdx                                      #34.50 c17
        movslq    %r10d, %r10                                   #30.34 c17
        cmpq      $1000, %r12                                   #36.14 c19
        jne       ..B1.13       # Prob 50%                      #36.14 c21
                                # LOE rdx rcx rbx rdi r9 r10 r11 r12 eax
..B1.11:                        # Preds ..B1.10
                                # Execution count [5.00e-01]
        xorl      %r10d, %r10d                                  #37.8 c1
        movslq    %eax, %r11                                    #38.21 c1
        movq      %rcx, %rbx                                    #40.8 c3
        movl      $16, %edx                                     #41.8 c3
                                # LOE rdx rbx rdi r9 r10 r11 r12
..B1.13:                        # Preds ..B1.10 ..B1.11
                                # Execution count [9.00e-01]
        lea       (%rdi,%r12,8), %r14                           #248.42 c1
        movq      $0, 8(%rsp)                                   #37.8[spill] c1
        movq      %r12, %r8                                     #250.42 c1
        movq      %r14, 232(%rsp)                               #248.42[spill] c3
        lea       (%r12,%r12,2), %r14                           #102.78 c3
        movq      %rbx, 56(%rsp)                                #119.57[spill] c5
        shlq      $4, %r8                                       #250.42 c5
        movq      %rdi, 168(%rsp)                               #119.57[spill] c5
        movq      %r9, %r13                                     #65.61 c5
        movq      %r8, 824(%rsp)                                #250.42[spill] c7
        lea       (,%r14,4), %rax                               #102.78 c7
        movq      %rax, 760(%rsp)                               #102.78[spill] c9
        lea       (%rdi,%r8), %r15                              #250.42 c9
        movq      %r15, 240(%rsp)                               #250.42[spill] c11
        addq      %rdi, %rax                                    #249.42 c11
        movq      %rax, 264(%rsp)                               #249.42[spill] c13
        lea       (%rdi,%r14,8), %r15                           #252.42 c13
        movq      %r15, 224(%rsp)                               #252.42[spill] c15
        lea       (%r12,%r12,4), %rax                           #251.42 c15
        movq      %r12, %r15                                    #253.42 c15
        shlq      $7, %r13                                      #65.61 c15
        movq      %r13, 160(%rsp)                               #65.61[spill] c17
        lea       (%rdi,%rax,4), %r8                            #251.42 c17
        movq      %r8, 216(%rsp)                                #251.42[spill] c19
        shlq      $5, %r15                                      #253.42 c19
        movq      %r15, 856(%rsp)                               #253.42[spill] c21
        lea       (,%r12,4), %rcx                               #97.76 c21
        movq      %r15, %r8                                     #253.42 c21
        movq      %r9, %r13                                     #73.59 c21
        vpxord    %zmm6, %zmm6, %zmm6                           #74.35 c21
        subq      %rcx, %r8                                     #253.42 c23
        movq      %r8, 832(%rsp)                                #253.42[spill] c25
        addq      %rdi, %r8                                     #253.42 c25
        movq      %r8, 248(%rsp)                                #253.42[spill] c27
        addq      %rdi, %r15                                    #254.42 c27
        movq      %r15, 256(%rsp)                               #254.42[spill] c29
        lea       (%rcx,%rcx,8), %r8                            #255.42 c29
        movq      %r8, 776(%rsp)                                #255.42[spill] c31
        lea       (%rdi,%r8), %r15                              #255.42 c31
        movq      %r15, 272(%rsp)                               #255.42[spill] c33
        imulq     $44, %r12, %r15                               #257.42 c33
        movq      %r15, 784(%rsp)                               #257.42[spill] c37 stall 1
        lea       (%rdi,%rax,8), %r8                            #256.42 c37
        movq      %r8, 280(%rsp)                                #256.42[spill] c39
        lea       (%rdi,%r15), %r8                              #257.42 c39
        movq      %r8, 328(%rsp)                                #257.42[spill] c41
        movq      %r14, %r15                                    #258.42 c41
        lea       (%rdi,%r12,4), %rsi                           #247.42 c41
        movq      %rsi, 208(%rsp)                               #247.42[spill] c43
        shlq      $4, %r15                                      #258.42 c43
        movq      %r15, 840(%rsp)                               #258.42[spill] c45
        lea       (%rdi,%r15), %r8                              #258.42 c45
        movq      %r8, 344(%rsp)                                #258.42[spill] c47
        imulq     $52, %r12, %r15                               #259.42 c47
        movq      %r15, 768(%rsp)                               #259.42[spill] c51 stall 1
        lea       (%rdi,%r15), %r8                              #259.42 c51
        movq      %r8, 352(%rsp)                                #259.42[spill] c53
        lea       (,%r12,8), %r15                               #260.42 c53
        shlq      $6, %r13                                      #73.59 c53
        imulq     %rdx, %r11                                    #47.38 c53
        subq      %r12, %r15                                    #260.42 c55
        movq      %r15, 816(%rsp)                               #260.42[spill] c57
        lea       (%rdi,%r15,8), %r8                            #260.42 c57
        movq      %r8, 336(%rsp)                                #260.42[spill] c59
        movq      %r12, %r15                                    #261.42 c59
        lea       (,%r9,4), %rsi                                #96.76 c59
        imulq     %rbx, %r10                                    #56.42 c59
        shlq      $6, %r15                                      #261.42 c61
        subq      %rcx, %r15                                    #261.42 c63
        movq      %r15, 848(%rsp)                               #261.42[spill] c65
        lea       (%rdi,%r15), %r8                              #261.42 c65
        movq      %r8, 320(%rsp)                                #261.42[spill] c67
        movq      %r9, %r15                                     #108.57 c67
        lea       (%r9,%r9,2), %r8                              #107.57 c67
        movq      %r8, 384(%rsp)                                #107.57[spill] c69
        shlq      $4, %r15                                      #108.57 c69
        movq      %r15, 800(%rsp)                               #108.57[spill] c71
        lea       (%r9,%r9,4), %r15                             #109.57 c71
        movq      %r15, 312(%rsp)                               #109.57[spill] c73
        movq      %r9, %r15                                     #111.57 c73
        shlq      $4, %r8                                       #116.57 c73
        movq      %r8, 360(%rsp)                                #116.57[spill] c75
        shlq      $5, %r15                                      #111.57 c75
        movq      %r15, 376(%rsp)                               #111.57[spill] c77
        subq      %rsi, %r15                                    #111.57 c77
        movq      %r15, 368(%rsp)                               #111.57[spill] c79
        lea       (%rsi,%rsi,8), %r15                           #113.57 c79
        movq      %r15, 304(%rsp)                               #113.57[spill] c81
        imulq     $44, %r9, %r15                                #115.57 c81
        movq      %r15, 296(%rsp)                               #115.57[spill] c85 stall 1
        imulq     $52, %r9, %r8                                 #117.57 c85
        movq      %r8, 288(%rsp)                                #117.57[spill] c89 stall 1
        lea       (,%r9,8), %r15                                #118.57 c89
        movq      %r13, %r8                                     #119.57 c89
        subq      %r9, %r15                                     #118.57 c91
        movq      %r15, 808(%rsp)                               #118.57[spill] c93
        subq      %rsi, %r8                                     #119.57 c93
        movq      %r8, 792(%rsp)                                #119.57[spill] c95
        movq      8(%rsp), %r15                                 #119.57[spill] c95
                                # LOE rax rdx rcx rsi r9 r10 r11 r12 r13 r14 r15 zmm6
..B1.14:                        # Preds ..B1.52 ..B1.13
                                # Execution count [2.62e+00]
        cmpq      %r12, %r11                                    #48.18 c1
        jae       ..B1.54       # Prob 20%                      #48.18 c3
                                # LOE rax rdx rcx rsi r9 r10 r11 r12 r13 r14 r15 zmm6
..B1.15:                        # Preds ..B1.14
                                # Execution count [2.09e+00]
        xorl      %edi, %edi                                    #51.36 c1
        xorl      %r8d, %r8d                                    #51.36 c1
        testq     %r9, %r9                                      #51.36 c3
        jbe       ..B1.52       # Prob 10%                      #51.36 c5
                                # LOE rax rdx rcx rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm6
..B1.16:                        # Preds ..B1.15
                                # Execution count [7.89e-01]
        lea       16(%r11), %rbx                                #57.46 c1
        movq      %rax, 712(%rsp)                               #57.46[spill] c1
        movq      %r14, 720(%rsp)                               #57.46[spill] c3
        movq      %r13, 560(%rsp)                               #57.46[spill] c3
        movq      %rcx, 728(%rsp)                               #57.46[spill] c5
        movq      %r15, 8(%rsp)                                 #57.46[spill] c5
        movq      %rdx, (%rsp)                                  #57.46[spill] c7
        movq      56(%rsp), %rdx                                #57.46[spill] c7
                                # LOE rdx rbx rsi rdi r8 r9 r10 r11 r12 zmm6
..B1.17:                        # Preds ..B1.50 ..B1.16
                                # Execution count [4.38e+00]
        xorl      %ecx, %ecx                                    #55.29 c1
        lea       2000(%rdi), %rax                              #61.48 c1
        testq     %rdx, %rdx                                    #55.42 c1
        jbe       ..B1.50       # Prob 10%                      #55.42 c3
                                # LOE rax rdx rcx rbx rsi rdi r8 r9 r10 r11 r12 zmm6
..B1.18:                        # Preds ..B1.17
                                # Execution count [3.94e+00]
        movq      %r9, 704(%rsp)                                #[spill] c1
                                # LOE rax rdx rcx rbx rsi rdi r8 r10 r11 r12 zmm6
..B1.19:                        # Preds ..B1.18 ..B1.48
                                # Execution count [2.19e+01]
        movq      %r11, %r9                                     #57.30 c1
        movq      %r11, %r14                                    #57.30 c1
        negq      %r9                                           #57.30 c3
        cmpq      %rbx, %r11                                    #57.46 c3
        jae       ..B1.48       # Prob 10%                      #57.46 c5
                                # LOE rax rdx rcx rbx rsi rdi r8 r9 r10 r11 r12 r14 zmm6
..B1.20:                        # Preds ..B1.19
                                # Execution count [1.97e+01]
        lea       (%r10,%rcx), %r13                             #56.55 c1
        movq      %rcx, 64(%rsp)                                #65.50[spill] c1
        lea       32(%r13), %r15                                #65.50 c3
        movq      %r10, 72(%rsp)                                #65.50[spill] c3
        movq      704(%rsp), %rcx                               #65.50[spill] c5
        movq      %r11, 80(%rsp)                                #65.50[spill] c5
        movq      %rdx, 56(%rsp)                                #65.50[spill] c7
                                # LOE rax rcx rbx rsi rdi r8 r9 r12 r13 r14 r15 zmm6
..B1.21:                        # Preds ..B1.46 ..B1.20
                                # Execution count [5.73e+01]
        cmpq      %r12, %r14                                    #58.23 c1
        jae       ..B1.47       # Prob 20%                      #58.23 c3
                                # LOE rax rcx rbx rsi rdi r8 r9 r12 r13 r14 r15 zmm6
..B1.22:                        # Preds ..B1.21
                                # Execution count [4.58e+01]
        movq      %r8, %rdx                                     #61.32 c1
        movq      %rdi, %r10                                    #61.32 c1
        lea       16(%r14), %r11                                #68.48 c1
        cmpq      %rax, %rdi                                    #61.48 c3
        jae       ..B1.46       # Prob 10%                      #61.48 c5
                                # LOE rax rdx rcx rbx rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm6
..B1.23:                        # Preds ..B1.22
                                # Execution count [4.13e+01]
        movq      %r11, 408(%rsp)                               #[spill] c1
        movq      %r9, 144(%rsp)                                #[spill] c1
        movq      %r14, 152(%rsp)                               #[spill] c3
        movq      432(%rsp), %r14                               #[spill] c3
        movq      %rbx, 88(%rsp)                                #[spill] c5
        movq      %r8, 96(%rsp)                                 #[spill] c7
        movq      %rdi, 104(%rsp)                               #[spill] c7
        movq      %r12, 744(%rsp)                               #[spill] c9
                                # LOE rax rdx rcx rsi r10 r13 r14 r15 zmm6
..B1.24:                        # Preds ..B1.23 ..B1.44
                                # Execution count [1.20e+02]
        cmpq      %rcx, %r10                                    #62.25 c1
        jae       ..B1.45       # Prob 20%                      #62.25 c3
                                # LOE rax rdx rcx rsi r10 r13 r14 r15 zmm6
..B1.25:                        # Preds ..B1.24
                                # Execution count [9.60e+01]
        movq      %rsi, %r8                                     #65.34 c1
        movq      %r13, %r9                                     #65.34 c1
        lea       2000(%r10), %r12                              #93.30 c1
        imulq     %r13, %r8                                     #65.34 c3
        cmpq      %r15, %r13                                    #65.50 c3
        jae       ..B1.44       # Prob 10%                      #65.50 c5
                                # LOE rax rdx rcx rsi r8 r9 r10 r12 r13 r14 r15 zmm6
..B1.26:                        # Preds ..B1.25
                                # Execution count [8.64e+01]
        movq      728(%rsp), %r11                               #97.60[spill] c1
        movq      %rdx, 424(%rsp)                               #96.60[spill] c1
        lea       4(%r10), %rdi                                 #94.23 c3
        movq      %rdi, 528(%rsp)                               #96.60[spill] c5
        imulq     %r10, %r11                                    #97.60 c5
        movq      %r12, 736(%rsp)                               #96.60[spill] c5
        lea       (%r14,%r10,4), %rbx                           #96.60 c7
        movq      %r10, 536(%rsp)                               #96.60[spill] c7
        addq      392(%rsp), %r11                               #97.60[spill] c9
        movq      %rbx, 520(%rsp)                               #96.60[spill] c9
        movq      408(%rsp), %r10                               #96.60[spill] c11
        movq      %r11, 184(%rsp)                               #96.60[spill] c11
        movq      144(%rsp), %r11                               #96.60[spill] c13
        movq      %r13, 112(%rsp)                               #96.60[spill] c15
        movq      152(%rsp), %r12                               #96.60[spill] c17
        movq      %rax, 120(%rsp)                               #96.60[spill] c17
        movq      160(%rsp), %r13                               #96.60[spill] c19
        movq      %rsi, 176(%rsp)                               #96.60[spill] c21
                                # LOE r8 r9 r10 r11 r12 r13 r15 zmm6
..B1.27:                        # Preds ..B1.42 ..B1.26
                                # Execution count [4.80e+02]
        movl      $65535, %esi                                  #67.33 c1
        movq      %r11, %rdi                                    #68.35 c1
        lea       32(%r9), %rdx                                 #73.50 c1
        movq      %r12, %rbx                                    #68.35 c3
        cmpq      %r10, %r12                                    #68.48 c3
        jae       ..B1.56       # Prob 10%                      #68.48 c5
                                # LOE rdx rbx rdi r8 r9 r10 r11 r12 r13 r15 esi zmm6
..B1.28:                        # Preds ..B1.27
                                # Execution count [4.32e+02]
        lea       (%r13,%r8), %rax                              #73.50 c1
        movq      %rdx, 568(%rsp)                               #70.21[spill] c1
        movl      $1, %ecx                                      #70.21 c1
        movq      %rax, 128(%rsp)                               #70.21[spill] c3
        movq      %r8, 400(%rsp)                                #70.21[spill] c3
        movq      %r9, 416(%rsp)                                #70.21[spill] c5
        movq      %r15, 136(%rsp)                               #70.21[spill] c5
                                # LOE rbx rdi ecx esi zmm6
..B1.29:                        # Preds ..B1.40 ..B1.28
                                # Execution count [2.40e+03]
        movq      744(%rsp), %rax                               #25.1[spill] c1
        lea       16(%rbx), %r14                                #69.28 c1
        lea       (%rax,%rdi), %rdx                             #25.1 c5 stall 1
        shlx      %edx, %ecx, %r8d                              #70.41 c5
        movq      400(%rsp), %rdx                               #73.37[spill] c5
        addl      $-1, %r8d                                     #70.47 c7
        cmpq      %rax, %r14                                    #70.21 c7
        movq      416(%rsp), %rax                               #73.50[spill] c7
        cmovae    %r8d, %esi                                    #70.21 c9
        cmpq      568(%rsp), %rax                               #73.50[spill] c11
        jae       ..B1.40       # Prob 10%                      #73.50 c13
                                # LOE rax rdx rbx rdi r14 ecx esi zmm6
..B1.30:                        # Preds ..B1.29
                                # Execution count [2.16e+03]
        movq      184(%rsp), %r13                               #97.60[spill] c1
        movq      %r14, 200(%rsp)                               #261.42[spill] c1
        vmovaps   %zmm6, %zmm7                                  #76.43 c1
        movq      %rdi, 192(%rsp)                               #261.42[spill] c3
        movq      168(%rsp), %r9                                #246.42[spill] c5
        movl      %esi, 552(%rsp)                               #261.42[spill] c5
        lea       (%r13,%rbx,4), %r12                           #97.60 c7
        movq      %r12, 544(%rsp)                               #97.60[spill] c9
        lea       (%r9,%rbx,4), %r8                             #246.42 c9
        movq      %r8, 592(%rsp)                                #246.42[spill] c11
        movq      392(%rsp), %r11                               #140.62[spill] c11
        movq      208(%rsp), %r15                               #247.42[spill] c13
        movq      232(%rsp), %r12                               #248.42[spill] c15
        movq      240(%rsp), %r8                                #250.42[spill] c17
        lea       (%r11,%rbx,4), %r10                           #140.62 c19
        movq      %r10, 752(%rsp)                               #140.62[spill] c21
        lea       (%r15,%rbx,4), %r13                           #247.42 c21
        movq      %r13, 600(%rsp)                               #247.42[spill] c23
        lea       (%r12,%rbx,4), %r11                           #248.42 c23
        movq      %r11, 608(%rsp)                               #248.42[spill] c25
        lea       (%r8,%rbx,4), %r15                            #250.42 c25
        movq      %r15, 624(%rsp)                               #250.42[spill] c27
        movq      264(%rsp), %r10                               #249.42[spill] c27
        movq      224(%rsp), %r11                               #252.42[spill] c29
        movq      272(%rsp), %r15                               #255.42[spill] c31
        lea       (%r10,%rbx,4), %r9                            #249.42 c33
        movq      %r9, 616(%rsp)                                #249.42[spill] c35
        lea       (%r11,%rbx,4), %r10                           #252.42 c35
        movq      %r10, 640(%rsp)                               #252.42[spill] c37
        lea       (%r15,%rbx,4), %r11                           #255.42 c37
        movq      %r11, 680(%rsp)                               #261.42[spill] c39
        movq      352(%rsp), %r15                               #259.42[spill] c39
        movq      216(%rsp), %r13                               #251.42[spill] c41
        lea       (%r15,%rbx,4), %r15                           #259.42 c43
        movq      %r15, 584(%rsp)                               #259.42[spill] c45
        movq      336(%rsp), %r15                               #260.42[spill] c45
        movq      248(%rsp), %r9                                #253.42[spill] c47
        movq      256(%rsp), %r8                                #254.42[spill] c49
        lea       (%r13,%rbx,4), %r12                           #251.42 c51
        movq      %r12, 632(%rsp)                               #251.42[spill] c53
        lea       (%r15,%rbx,4), %r15                           #260.42 c53
        movq      %r15, 576(%rsp)                               #260.42[spill] c55
        lea       (%r9,%rbx,4), %r13                            #253.42 c55
        movq      %r13, 648(%rsp)                               #261.42[spill] c57
        lea       (%r8,%rbx,4), %r12                            #254.42 c57
        movq      %r12, 656(%rsp)                               #261.42[spill] c59
        movq      280(%rsp), %r10                               #256.42[spill] c59
        movq      328(%rsp), %r9                                #257.42[spill] c61
        movq      344(%rsp), %r8                                #258.42[spill] c63
        movq      320(%rsp), %r15                               #261.42[spill] c65
        lea       (%r10,%rbx,4), %r10                           #256.42 c67
        movq      %r10, 696(%rsp)                               #261.42[spill] c69
        lea       (%r9,%rbx,4), %r9                             #257.42 c69
        movq      %r9, 672(%rsp)                                #261.42[spill] c71
        lea       (%r8,%rbx,4), %r8                             #258.42 c71
        movq      %r8, 664(%rsp)                                #261.42[spill] c73
        lea       (%r15,%rbx,4), %rbx                           #261.42 c73
        movq      %rbx, 688(%rsp)                               #261.42[spill] c75
        movq      288(%rsp), %r15                               #261.42[spill] c75
        movq      296(%rsp), %r8                                #261.42[spill] c77
        movq      360(%rsp), %rdi                               #261.42[spill] c79
        movq      304(%rsp), %r9                                #261.42[spill] c81
        movq      368(%rsp), %r10                               #261.42[spill] c83
        movq      376(%rsp), %r11                               #261.42[spill] c85
        movq      312(%rsp), %r12                               #261.42[spill] c87
        movq      384(%rsp), %r13                               #261.42[spill] c89
        movq      176(%rsp), %r14                               #261.42[spill] c91
        movq      704(%rsp), %rcx                               #261.42[spill] c93
                                # LOE rax rdx rcx rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm6 zmm7
..B1.31:                        # Preds ..B1.38 ..B1.30
                                # Execution count [1.20e+04]
        vmovaps   %zmm6, %zmm4                                  #91.43 c1
        vmovaps   %zmm7, %zmm18                                 #76.43 c1
        vmovaps   %zmm6, %zmm19                                 #77.43 c3
        vmovaps   %zmm6, %zmm20                                 #78.43 c3
        vmovaps   %zmm6, %zmm21                                 #79.43 c5
        vmovaps   %zmm6, %zmm22                                 #80.43 c5
        vmovaps   %zmm6, %zmm23                                 #81.43 c7
        vmovaps   %zmm6, %zmm24                                 #82.43 c7
        vmovaps   %zmm6, %zmm25                                 #83.43 c9
        vmovaps   %zmm6, %zmm26                                 #84.43 c9
        vmovaps   %zmm6, %zmm27                                 #85.43 c11
        vmovaps   %zmm6, %zmm28                                 #86.43 c11
        vmovaps   %zmm6, %zmm29                                 #87.43 c13
        vmovaps   %zmm6, %zmm30                                 #88.43 c13
        vmovaps   %zmm6, %zmm31                                 #89.43 c15
        vmovaps   %zmm6, %zmm5                                  #90.43 c15
        vmovaps   %zmm4, %zmm7                                  #91.43 c17
        cmpq      736(%rsp), %rcx                               #93.35[spill] c17
        jae       ..B1.34       # Prob 50%                      #93.35 c19
                                # LOE rax rdx rcx rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm4 zmm5 zmm6 zmm7 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31
..B1.32:                        # Preds ..B1.31
                                # Execution count [6.00e+03]
        cmpq      528(%rsp), %rcx                               #94.23[spill] c1
        jne       ..B1.55       # Prob 0%                       #94.23 c3
                                # LOE rax rdx rcx rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm6 zmm7
..B1.33:                        # Preds ..B1.32
                                # Execution count [6.00e+03]
        movl      552(%rsp), %ebx                               #99.45[spill] c1
        movq      544(%rsp), %rsi                               #99.45[spill] c1
        vmovaps   %zmm6, %zmm22                                 #125.38 c1
        vmovaps   %zmm6, %zmm5                                  #135.38 c1
        vmovaps   %zmm6, %zmm18                                 #121.38 c3
        vmovaps   %zmm6, %zmm19                                 #122.38 c3
        kmovw     %ebx, %k1                                     #99.45 c5
        movq      728(%rsp), %rbx                               #100.45[spill] c5
        vmovaps   %zmm6, %zmm20                                 #123.38 c5
        vmovaps   %zmm6, %zmm21                                 #124.38 c5
        vmovups   (%rsi), %zmm0{%k1}{z}                         #99.45 c7
        vmovaps   %zmm6, %zmm23                                 #126.38 c7
        vmovaps   %zmm6, %zmm24                                 #127.38 c7
        vmovups   (%rsi,%rbx), %zmm1{%k1}{z}                    #100.45 c9
        vmovaps   %zmm6, %zmm25                                 #128.38 c9
        vmovaps   %zmm6, %zmm26                                 #129.38 c9
        vmovaps   %zmm6, %zmm27                                 #130.38 c11
        vmovaps   %zmm6, %zmm28                                 #131.38 c11
        movq      744(%rsp), %rbx                               #101.45[spill] c13
        vmovaps   %zmm6, %zmm29                                 #132.38 c13
        vmovaps   %zmm6, %zmm30                                 #133.38 c13
        vmovaps   %zmm6, %zmm31                                 #134.38 c15
        vmovaps   %zmm6, %zmm4                                  #136.38 c15
        vmovups   (%rsi,%rbx,8), %zmm2{%k1}{z}                  #101.45 c17
        movq      760(%rsp), %rbx                               #102.45[spill] c17
        vmovups   (%rbx,%rsi), %zmm3{%k1}{z}                    #102.45 c21 stall 1
        movq      520(%rsp), %rsi                               #96.60[spill] c23
        lea       (%rsi,%rdx), %rbx                             #96.60 c27 stall 1
        movq      800(%rsp), %rsi                               #125.38[spill] c27
        v4fmaddps (%rbx), %zmm0, %zmm18                         #121.38 c29
        v4fmaddps (%rsi,%rbx), %zmm0, %zmm22                    #125.38 c31
        movq      808(%rsp), %rsi                               #135.38[spill] c31
        v4fmaddps (%rbx,%r14), %zmm0, %zmm19                    #122.38 c33
        v4fmaddps (%rbx,%rsi,8), %zmm0, %zmm5                   #135.38 c35
        movq      792(%rsp), %rsi                               #136.38[spill] c35
        v4fmaddps (%rbx,%rcx,8), %zmm0, %zmm20                  #123.38 c37
        v4fmaddps (%rbx,%r13,4), %zmm0, %zmm21                  #124.38 c39
        v4fmaddps (%rbx,%r12,4), %zmm0, %zmm23                  #126.38 c39
        v4fmaddps (%rbx,%r13,8), %zmm0, %zmm24                  #127.38 c41
        v4fmaddps (%r10,%rbx), %zmm0, %zmm25                    #128.38 c41
        v4fmaddps (%r11,%rbx), %zmm0, %zmm26                    #129.38 c43
        v4fmaddps (%r9,%rbx), %zmm0, %zmm27                     #130.38 c43
        v4fmaddps (%rbx,%r12,8), %zmm0, %zmm28                  #131.38 c45
        v4fmaddps (%r8,%rbx), %zmm0, %zmm29                     #132.38 c45
        v4fmaddps (%rdi,%rbx), %zmm0, %zmm30                    #133.38 c47
        v4fmaddps (%r15,%rbx), %zmm0, %zmm31                    #134.38 c47
        v4fmaddps (%rsi,%rbx), %zmm0, %zmm4                     #136.38 c49
        jmp       ..B1.38       # Prob 100%                     #136.38 c49
                                # LOE rax rdx rcx rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm4 zmm5 zmm6 zmm7 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31 k1
..B1.34:                        # Preds ..B1.31
                                # Execution count [6.00e+03]
        movq      736(%rsp), %rbx                               #138.54[spill] c1
        movq      536(%rsp), %rsi                               #138.41[spill] c1
        cmpq      536(%rsp), %rbx                               #138.54[spill] c5 stall 1
        jbe       ..B1.67       # Prob 10%                      #138.54 c7
                                # LOE rax rdx rcx rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm4 zmm5 zmm6 zmm7 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31
..B1.35:                        # Preds ..B1.34
                                # Execution count [5.40e+03]
        movl      552(%rsp), %ebx                               #139.62[spill] c1
        movq      %rax, 512(%rsp)                               #139.62[spill] c1
        kmovw     %ebx, %k1                                     #139.62 c5 stall 1
        movq      %rdx, 440(%rsp)                               #139.62[spill] c5
        movq      432(%rsp), %rbx                               #139.62[spill] c5
        addq      %rdx, %rbx                                    #139.62 c9 stall 1
        addq      424(%rsp), %rbx                               #139.62[spill] c11
                                # LOE rbx rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm4 zmm5 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31 k1
..B1.36:                        # Preds ..B1.36 ..B1.35
                                # Execution count [3.00e+04]
        movq      728(%rsp), %rcx                               #140.62[spill] c1
        movq      744(%rsp), %rax                               #144.47[spill] c1
        movq      %rcx, %rdx                                    #140.62 c5 stall 1
        imulq     %rsi, %rdx                                    #140.62 c7
        addq      $16, %rsi                                     #138.58 c7
        movq      %rsi, 864(%rsp)                               #138.58[spill] c9
        movq      816(%rsp), %rsi                               #156.47[spill] c9
        addq      752(%rsp), %rdx                               #140.62[spill] c11
        vmovups   (%rdx,%rcx), %zmm1{%k1}{z}                    #143.47 c13
        movq      760(%rsp), %rcx                               #145.47[spill] c13
        vmovups   (%rcx,%rdx), %zmm6{%k1}{z}                    #145.47 c17 stall 1
        vmovups   %zmm6, 448(%rsp)                              #145.47[spill] c23 stall 2
        vmovups   (%rdx,%rax,8), %zmm2{%k1}{z}                  #144.47 c23
        movq      824(%rsp), %rax                               #146.47[spill] c29 stall 2
        movq      720(%rsp), %rcx                               #148.47[spill] c29
        vmovups   (%rax,%rdx), %zmm6{%k1}{z}                    #146.47 c33 stall 1
        movq      712(%rsp), %rax                               #147.47[spill] c33
        vmovups   (%rdx,%rcx,8), %zmm8{%k1}{z}                  #148.47 c37 stall 1
        movq      832(%rsp), %rcx                               #149.47[spill] c39
        vmovups   (%rdx,%rax,4), %zmm7{%k1}{z}                  #147.47 c43 stall 1
        vmovups   (%rcx,%rdx), %zmm9{%k1}{z}                    #149.47 c43
        movq      856(%rsp), %rcx                               #150.47[spill] c49 stall 2
        vmovups   (%rdx,%rax,8), %zmm12{%k1}{z}                 #152.47 c49
        movq      784(%rsp), %rax                               #153.47[spill] c53 stall 1
        vmovups   (%rcx,%rdx), %zmm10{%k1}{z}                   #150.47 c55
        movq      776(%rsp), %rcx                               #151.47[spill] c57
        vmovups   (%rax,%rdx), %zmm13{%k1}{z}                   #153.47 c61 stall 1
        movq      840(%rsp), %rax                               #154.47[spill] c61
        vmovups   (%rcx,%rdx), %zmm11{%k1}{z}                   #151.47 c65 stall 1
        vmovups   (%rax,%rdx), %zmm14{%k1}{z}                   #154.47 c67
        movq      768(%rsp), %rcx                               #155.47[spill] c71 stall 1
        movq      848(%rsp), %rax                               #157.47[spill] c73
        vmovups   (%rdx), %zmm0{%k1}{z}                         #142.47 c75
        vmovups   448(%rsp), %zmm3                              #176.40[spill] c77
        vmovups   (%rcx,%rdx), %zmm15{%k1}{z}                   #155.47 c81 stall 1
        vmovups   (%rdx,%rsi,8), %zmm16{%k1}{z}                 #156.47 c83
        vmovups   (%rax,%rdx), %zmm17{%k1}{z}                   #157.47 c87 stall 1
        movq      704(%rsp), %rdx                               #178.40[spill] c89
        movq      800(%rsp), %rcx                               #180.40[spill] c93 stall 1
        movq      808(%rsp), %rsi                               #190.40[spill] c93
        movq      792(%rsp), %rax                               #191.40[spill] c97 stall 1
        v4fmaddps (%rbx), %zmm0, %zmm18                         #176.40 c97
        v4fmaddps (%rbx,%r14), %zmm0, %zmm19                    #177.40 c99
        v4fmaddps (%rbx,%rdx,8), %zmm0, %zmm20                  #178.40 c101
        v4fmaddps (%rbx,%r13,4), %zmm0, %zmm21                  #179.40 c101
        v4fmaddps (%rbx,%rcx), %zmm0, %zmm22                    #180.40 c103
        v4fmaddps (%rbx,%r12,4), %zmm0, %zmm23                  #181.40 c103
        v4fmaddps (%rbx,%r13,8), %zmm0, %zmm24                  #182.40 c105
        v4fmaddps (%rbx,%r10), %zmm0, %zmm25                    #183.40 c105
        v4fmaddps (%rbx,%r11), %zmm0, %zmm26                    #184.40 c107
        .byte     144                                           #185.40 c107
        v4fmaddps (%rbx,%r9), %zmm0, %zmm27                     #185.40 c107
        v4fmaddps (%rbx,%r12,8), %zmm0, %zmm28                  #186.40 c109
        .byte     102                                           #187.40 c109
        .byte     144                                           #187.40 c109
        v4fmaddps (%rbx,%r8), %zmm0, %zmm29                     #187.40 c109
        v4fmaddps (%rbx,%rdi), %zmm0, %zmm30                    #188.40 c111
        v4fmaddps (%rbx,%r15), %zmm0, %zmm31                    #189.40 c111
        v4fmaddps (%rbx,%rsi,8), %zmm0, %zmm5                   #190.40 c113
        v4fmaddps (%rbx,%rax), %zmm0, %zmm4                     #191.40 c113
        vmovaps   %zmm6, %zmm0                                  #193.40 c113
        vmovaps   %zmm7, %zmm1                                  #193.40 c113
        vmovaps   %zmm8, %zmm2                                  #193.40 c115
        vmovaps   %zmm9, %zmm3                                  #193.40 c115
        v4fmaddps 16(%rbx,%rax), %zmm0, %zmm4                   #208.40 c117
        v4fmaddps 16(%rbx), %zmm0, %zmm18                       #193.40 c117
        v4fmaddps 16(%rbx,%r14), %zmm0, %zmm19                  #194.40 c119
        v4fmaddps 16(%rbx,%rdx,8), %zmm0, %zmm20                #195.40 c119
        v4fmaddps 16(%rbx,%r13,4), %zmm0, %zmm21                #196.40 c121
        v4fmaddps 16(%rbx,%rcx), %zmm0, %zmm22                  #197.40 c121
        v4fmaddps 16(%rbx,%r12,4), %zmm0, %zmm23                #198.40 c123
        v4fmaddps 16(%rbx,%r13,8), %zmm0, %zmm24                #199.40 c123
        v4fmaddps 16(%rbx,%r10), %zmm0, %zmm25                  #200.40 c125
        v4fmaddps 16(%rbx,%r11), %zmm0, %zmm26                  #201.40 c125
        .byte     15                                            #202.40 c127
        .byte     31                                            #202.40 c127
        .byte     64                                            #202.40 c127
        .byte     0                                             #202.40 c127
        v4fmaddps 16(%rbx,%r9), %zmm0, %zmm27                   #202.40 c127
        v4fmaddps 16(%rbx,%r12,8), %zmm0, %zmm28                #203.40 c127
        v4fmaddps 16(%rbx,%r8), %zmm0, %zmm29                   #204.40 c129
        v4fmaddps 16(%rbx,%rdi), %zmm0, %zmm30                  #205.40 c129
        v4fmaddps 16(%rbx,%r15), %zmm0, %zmm31                  #206.40 c131
        v4fmaddps 16(%rbx,%rsi,8), %zmm0, %zmm5                 #207.40 c131
        vmovaps   %zmm10, %zmm0                                 #210.40 c131
        vmovaps   %zmm11, %zmm1                                 #210.40 c131
        vmovaps   %zmm12, %zmm2                                 #210.40 c133
        vmovaps   %zmm13, %zmm3                                 #210.40 c133
        v4fmaddps 32(%rbx,%rsi,8), %zmm0, %zmm5                 #224.40 c135
        v4fmaddps 32(%rbx), %zmm0, %zmm18                       #210.40 c135
        v4fmaddps 32(%rbx,%r14), %zmm0, %zmm19                  #211.40 c137
        v4fmaddps 32(%rbx,%rdx,8), %zmm0, %zmm20                #212.40 c137
        v4fmaddps 32(%rbx,%r13,4), %zmm0, %zmm21                #213.40 c139
        v4fmaddps 32(%rbx,%rcx), %zmm0, %zmm22                  #214.40 c139
        .byte     15                                            #215.40 c141
        .byte     31                                            #215.40 c141
        .byte     64                                            #215.40 c141
        .byte     0                                             #215.40 c141
        v4fmaddps 32(%rbx,%r12,4), %zmm0, %zmm23                #215.40 c141
        v4fmaddps 32(%rbx,%r13,8), %zmm0, %zmm24                #216.40 c141
        v4fmaddps 32(%rbx,%r10), %zmm0, %zmm25                  #217.40 c143
        v4fmaddps 32(%rbx,%r11), %zmm0, %zmm26                  #218.40 c143
        v4fmaddps 32(%rbx,%r9), %zmm0, %zmm27                   #219.40 c145
        v4fmaddps 32(%rbx,%r12,8), %zmm0, %zmm28                #220.40 c145
        .byte     15                                            #221.40 c147
        .byte     31                                            #221.40 c147
        .byte     64                                            #221.40 c147
        .byte     0                                             #221.40 c147
        v4fmaddps 32(%rbx,%r8), %zmm0, %zmm29                   #221.40 c147
        v4fmaddps 32(%rbx,%rdi), %zmm0, %zmm30                  #222.40 c147
        v4fmaddps 32(%rbx,%r15), %zmm0, %zmm31                  #223.40 c149
        v4fmaddps 32(%rbx,%rax), %zmm0, %zmm4                   #225.40 c149
        vmovaps   %zmm14, %zmm0                                 #227.40 c149
        vmovaps   %zmm15, %zmm1                                 #227.40 c149
        vmovaps   %zmm16, %zmm2                                 #227.40 c151
        vmovaps   %zmm17, %zmm3                                 #227.40 c151
        v4fmaddps 48(%rbx,%rsi,8), %zmm0, %zmm5                 #241.40 c153
        movq      864(%rsp), %rsi                               #138.54[spill] c153
        v4fmaddps 48(%rbx), %zmm0, %zmm18                       #227.40 c155
        v4fmaddps 48(%rbx,%r14), %zmm0, %zmm19                  #228.40 c157
        v4fmaddps 48(%rbx,%rdx,8), %zmm0, %zmm20                #229.40 c157
        v4fmaddps 48(%rbx,%r13,4), %zmm0, %zmm21                #230.40 c159
        v4fmaddps 48(%rbx,%rcx), %zmm0, %zmm22                  #231.40 c159
        v4fmaddps 48(%rbx,%r12,4), %zmm0, %zmm23                #232.40 c161
        v4fmaddps 48(%rbx,%r13,8), %zmm0, %zmm24                #233.40 c161
        v4fmaddps 48(%rbx,%r10), %zmm0, %zmm25                  #234.40 c163
        v4fmaddps 48(%rbx,%r11), %zmm0, %zmm26                  #235.40 c163
        .byte     102                                           #236.40 c165
        .byte     144                                           #236.40 c165
        v4fmaddps 48(%rbx,%r9), %zmm0, %zmm27                   #236.40 c165
        .byte     15                                            #237.40 c165
        .byte     31                                            #237.40 c165
        .byte     0                                             #237.40 c165
        v4fmaddps 48(%rbx,%r12,8), %zmm0, %zmm28                #237.40 c165
        .byte     144                                           #238.40 c167
        v4fmaddps 48(%rbx,%r8), %zmm0, %zmm29                   #238.40 c167
        v4fmaddps 48(%rbx,%rdi), %zmm0, %zmm30                  #239.40 c167
        v4fmaddps 48(%rbx,%r15), %zmm0, %zmm31                  #240.40 c169
        v4fmaddps 48(%rbx,%rax), %zmm0, %zmm4                   #242.40 c169
        addq      $64, %rbx                                     #138.58 c169
        cmpq      736(%rsp), %rsi                               #138.54[spill] c171
        jb        ..B1.36       # Prob 82%                      #138.54 c173
                                # LOE rdx rbx rsi rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm4 zmm5 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31 k1
..B1.37:                        # Preds ..B1.36
                                # Execution count [5.40e+03]
        movq      %rdx, %rcx                                    # c1
        vpxord    %zmm7, %zmm7, %zmm7                           # c1
        vpxord    %zmm6, %zmm6, %zmm6                           # c1
        movq      512(%rsp), %rax                               #[spill] c1
        movq      440(%rsp), %rdx                               #[spill] c1
                                # LOE rax rdx rcx rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm4 zmm5 zmm6 zmm7 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31 k1
..B1.38:                        # Preds ..B1.33 ..B1.37 ..B1.67
                                # Execution count [1.20e+04]
        movq      744(%rsp), %rbx                               #246.42[spill] c1
        movq      592(%rsp), %rsi                               #246.21[spill] c1
        imulq     %rax, %rbx                                    #246.42 c5 stall 1
        vmovups   %zmm18, (%rsi,%rbx,4){%k1}                    #246.21 c9 stall 1
        movq      600(%rsp), %rsi                               #247.21[spill] c9
        vmovups   %zmm19, (%rsi,%rbx,4){%k1}                    #247.21 c13 stall 1
        addq      $16, %rax                                     #73.54 c13
        movq      608(%rsp), %rsi                               #248.21[spill] c15
        vmovups   %zmm20, (%rsi,%rbx,4){%k1}                    #248.21 c19 stall 1
        movq      616(%rsp), %rsi                               #249.21[spill] c19
        vmovups   %zmm21, (%rsi,%rbx,4){%k1}                    #249.21 c23 stall 1
        movq      624(%rsp), %rsi                               #250.21[spill] c25
        vmovups   %zmm22, (%rsi,%rbx,4){%k1}                    #250.21 c29 stall 1
        movq      632(%rsp), %rsi                               #251.21[spill] c29
        vmovups   %zmm23, (%rsi,%rbx,4){%k1}                    #251.21 c33 stall 1
        movq      640(%rsp), %rsi                               #252.21[spill] c35
        vmovups   %zmm24, (%rsi,%rbx,4){%k1}                    #252.21 c39 stall 1
        movq      648(%rsp), %rsi                               #253.21[spill] c39
        vmovups   %zmm25, (%rsi,%rbx,4){%k1}                    #253.21 c43 stall 1
        movq      656(%rsp), %rsi                               #254.21[spill] c45
        vmovups   %zmm26, (%rsi,%rbx,4){%k1}                    #254.21 c49 stall 1
        movq      680(%rsp), %rsi                               #255.21[spill] c49
        vmovups   %zmm27, (%rsi,%rbx,4){%k1}                    #255.21 c53 stall 1
        movq      696(%rsp), %rsi                               #256.21[spill] c55
        vmovups   %zmm28, (%rsi,%rbx,4){%k1}                    #256.21 c59 stall 1
        movq      672(%rsp), %rsi                               #257.21[spill] c59
        vmovups   %zmm29, (%rsi,%rbx,4){%k1}                    #257.21 c63 stall 1
        movq      664(%rsp), %rsi                               #258.21[spill] c65
        vmovups   %zmm30, (%rsi,%rbx,4){%k1}                    #258.21 c69 stall 1
        movq      584(%rsp), %rsi                               #259.21[spill] c69
        vmovups   %zmm31, (%rsi,%rbx,4){%k1}                    #259.21 c73 stall 1
        movq      576(%rsp), %rsi                               #260.21[spill] c75
        vmovups   %zmm5, (%rsi,%rbx,4){%k1}                     #260.21 c79 stall 1
        movq      688(%rsp), %rsi                               #261.21[spill] c79
        vmovups   %zmm4, (%rsi,%rbx,4){%k1}                     #261.21 c83 stall 1
        addq      560(%rsp), %rdx                               #73.54[spill] c85
        cmpq      568(%rsp), %rax                               #73.50[spill] c87
        jb        ..B1.31       # Prob 82%                      #73.50 c89
                                # LOE rax rdx rcx rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm6 zmm7
..B1.39:                        # Preds ..B1.38
                                # Execution count [2.16e+03]
        movq      200(%rsp), %r14                               #[spill] c1
        movq      192(%rsp), %rdi                               #[spill] c1
        movl      $1, %ecx                                      # c1
        movl      552(%rsp), %esi                               #[spill] c5 stall 1
                                # LOE rdi r14 ecx esi zmm6
..B1.40:                        # Preds ..B1.39 ..B1.29
                                # Execution count [2.40e+03]
        addq      $-16, %rdi                                    #69.28 c1
        movq      %r14, %rbx                                    #68.52 c1
        cmpq      408(%rsp), %r14                               #68.48[spill] c1
        jb        ..B1.29       # Prob 82%                      #68.48 c3
                                # LOE rbx rdi ecx esi zmm6
..B1.41:                        # Preds ..B1.40
                                # Execution count [4.32e+02]
        movq      128(%rsp), %rax                               #[spill] c1
        movq      136(%rsp), %r15                               #[spill] c1
        movq      144(%rsp), %r11                               #[spill] c5 stall 1
        movq      152(%rsp), %r12                               #[spill] c5
        movq      160(%rsp), %r13                               #[spill] c9 stall 1
        movq      568(%rsp), %rdx                               #[spill] c9
        movq      408(%rsp), %r10                               #[spill] c13 stall 1
                                # LOE rax rdx r10 r11 r12 r13 r15 zmm6
..B1.42:                        # Preds ..B1.41 ..B1.56
                                # Execution count [4.80e+02]
        movq      %rax, %r8                                     #65.55 c1
        movq      %rdx, %r9                                     #65.55 c1
        cmpq      %r15, %rdx                                    #65.50 c3
        jb        ..B1.27       # Prob 82%                      #65.50 c5
                                # LOE r8 r9 r10 r11 r12 r13 r15 zmm6
..B1.43:                        # Preds ..B1.42
                                # Execution count [8.64e+01]
        movq      424(%rsp), %rdx                               #[spill] c1
        movq      432(%rsp), %r14                               #[spill] c1
        movq      736(%rsp), %r12                               #[spill] c5 stall 1
        movq      704(%rsp), %rcx                               #[spill] c5
        movq      112(%rsp), %r13                               #[spill] c9 stall 1
        movq      120(%rsp), %rax                               #[spill] c9
        movq      176(%rsp), %rsi                               #[spill] c13 stall 1
                                # LOE rax rdx rcx rsi r12 r13 r14 r15 zmm6
..B1.44:                        # Preds ..B1.25 ..B1.43
                                # Execution count [9.60e+01]
        addq      $8000, %rdx                                   #93.30 c1
        movq      %r12, %r10                                    #61.53 c1
        cmpq      %rax, %r12                                    #61.48 c3
        jb        ..B1.24       # Prob 82%                      #61.48 c5
                                # LOE rax rdx rcx rsi r10 r13 r14 r15 zmm6
..B1.45:                        # Preds ..B1.24 ..B1.44
                                # Execution count [1.73e+01]
        movq      408(%rsp), %r11                               #[spill] c1
        movq      144(%rsp), %r9                                #[spill] c1
        movq      88(%rsp), %rbx                                #[spill] c5 stall 1
        movq      96(%rsp), %r8                                 #[spill] c5
        movq      104(%rsp), %rdi                               #[spill] c9 stall 1
        movq      744(%rsp), %r12                               #[spill] c9
                                # LOE rax rcx rbx rsi rdi r8 r9 r11 r12 r13 r15 zmm6
..B1.46:                        # Preds ..B1.22 ..B1.45
                                # Execution count [4.58e+01]
        addq      $-16, %r9                                     #68.48 c1
        movq      %r11, %r14                                    #57.51 c1
        cmpq      %rbx, %r11                                    #57.46 c3
        jb        ..B1.21       # Prob 82%                      #57.46 c5
                                # LOE rax rcx rbx rsi rdi r8 r9 r12 r13 r14 r15 zmm6
..B1.47:                        # Preds ..B1.21 ..B1.46
                                # Execution count [8.25e+00]
        movq      64(%rsp), %rcx                                #[spill] c1
        movq      72(%rsp), %r10                                #[spill] c1
        movq      80(%rsp), %r11                                #[spill] c5 stall 1
        movq      56(%rsp), %rdx                                #[spill] c5
                                # LOE rax rdx rcx rbx rsi rdi r8 r10 r11 r12 zmm6
..B1.48:                        # Preds ..B1.47 ..B1.19
                                # Execution count [2.19e+01]
        addq      $32, %rcx                                     #55.57 c1
        cmpq      %rdx, %rcx                                    #55.42 c3
        jb        ..B1.19       # Prob 82%                      #55.42 c5
                                # LOE rax rdx rcx rbx rsi rdi r8 r10 r11 r12 zmm6
..B1.49:                        # Preds ..B1.48
                                # Execution count [3.94e+00]
        movq      704(%rsp), %r9                                #[spill] c1
                                # LOE rax rdx rbx rsi r8 r9 r10 r11 r12 zmm6
..B1.50:                        # Preds ..B1.17 ..B1.49
                                # Execution count [4.38e+00]
        addq      $8000, %r8                                    #61.48 c1
        movq      %rax, %rdi                                    #51.39 c1
        cmpq      %r9, %rax                                     #51.36 c3
        jb        ..B1.17       # Prob 82%                      #51.36 c5
                                # LOE rdx rbx rsi rdi r8 r9 r10 r11 r12 zmm6
..B1.51:                        # Preds ..B1.50
                                # Execution count [7.89e-01]
        movq      %rdx, 56(%rsp)                                #[spill] c1
        movq      712(%rsp), %rax                               #[spill] c1
        movq      720(%rsp), %r14                               #[spill] c3
        movq      728(%rsp), %rcx                               #[spill] c5
        movq      560(%rsp), %r13                               #[spill] c7
        movq      8(%rsp), %r15                                 #[spill] c9
        movq      (%rsp), %rdx                                  #[spill] c11
                                # LOE rax rdx rcx rsi r9 r10 r11 r12 r13 r14 r15 zmm6
..B1.52:                        # Preds ..B1.51 ..B1.15
                                # Execution count [2.09e+00]
        addq      $16, %r15                                     #46.53 c1
        addq      $16, %r11                                     #46.53 c1
        cmpq      %rdx, %r15                                    #46.38 c3
        jb        ..B1.14       # Prob 82%                      #46.38 c5
                                # LOE rax rdx rcx rsi r9 r10 r11 r12 r13 r14 r15 zmm6
..B1.54:                        # Preds ..B1.14 ..B1.52
                                # Execution count [0.00e+00]
        movq      16(%rsp), %r15                                #25.1[spill] c1
	.cfi_restore 15
        movq      24(%rsp), %r14                                #25.1[spill] c1
	.cfi_restore 14
        movq      32(%rsp), %r13                                #25.1[spill] c5 stall 1
	.cfi_restore 13
        movq      40(%rsp), %r12                                #25.1[spill] c5
	.cfi_restore 12
        movq      48(%rsp), %rbx                                #25.1[spill] c9 stall 1
	.cfi_restore 3
        movq      %rbp, %rsp                                    #25.1 c11
        popq      %rbp                                          #25.1
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret                                                     #25.1
	.cfi_def_cfa 6, 16
	.cfi_escape 0x10, 0x03, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xb0, 0xfc, 0xff, 0xff, 0x22
	.cfi_offset 6, -16
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xa8, 0xfc, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0xa0, 0xfc, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0e, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x98, 0xfc, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0f, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x90, 0xfc, 0xff, 0xff, 0x22
                                # LOE
..B1.55:                        # Preds ..B1.32
                                # Execution count [3.00e+01]: Infreq
        movl      $.L_2__STRING.0, %edi                         #94.23 c1
        movl      $.L_2__STRING.1, %esi                         #94.23 c1
        movl      $94, %edx                                     #94.23 c3
        movl      $__$U0, %ecx                                  #94.23 c3
#       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #94.23 c5
                                # LOE
..B1.56:                        # Preds ..B1.27
                                # Execution count [4.80e+01]: Infreq
        lea       (%r13,%r8), %rax                              #65.55 c1
        jmp       ..B1.42       # Prob 100%                     #65.55 c1
                                # LOE rax rdx r10 r11 r12 r13 r15 zmm6
..B1.67:                        # Preds ..B1.34
                                # Execution count [6.00e+02]: Infreq
        movl      552(%rsp), %ebx                               #99.45[spill] c1
        kmovw     %ebx, %k1                                     #99.45 c5 stall 1
        jmp       ..B1.38       # Prob 100%                     #99.45 c5 stall 1
        .align    16,0x90
                                # LOE rax rdx rcx rdi r8 r9 r10 r11 r12 r13 r14 r15 zmm4 zmm5 zmm6 zmm7 zmm18 zmm19 zmm20 zmm21 zmm22 zmm23 zmm24 zmm25 zmm26 zmm27 zmm28 zmm29 zmm30 zmm31 k1
	.cfi_endproc
# mark_end;
	.type	sgemm_opt,@function
	.size	sgemm_opt,.-sgemm_opt
	.data
	.align 4
	.align 4
.2.5_2_kmpc_loc_struct_pack.12:
	.long	0
	.long	2
	.long	0
	.long	0
	.quad	.2.5_2__kmpc_loc_pack.11
	.align 4
.2.5_2__kmpc_loc_pack.11:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	115
	.byte	103
	.byte	101
	.byte	109
	.byte	109
	.byte	95
	.byte	111
	.byte	112
	.byte	116
	.byte	59
	.byte	50
	.byte	53
	.byte	59
	.byte	50
	.byte	53
	.byte	59
	.byte	59
	.space 2, 0x00 	# pad
	.align 4
.2.5_2_kmpc_loc_struct_pack.20:
	.long	0
	.long	2
	.long	0
	.long	0
	.quad	.2.5_2__kmpc_loc_pack.19
	.align 4
.2.5_2__kmpc_loc_pack.19:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	115
	.byte	103
	.byte	101
	.byte	109
	.byte	109
	.byte	95
	.byte	111
	.byte	112
	.byte	116
	.byte	59
	.byte	50
	.byte	53
	.byte	59
	.byte	50
	.byte	55
	.byte	48
	.byte	59
	.byte	59
	.data
# -- End  sgemm_opt, L_sgemm_opt_25__par_region0_2.0
	.section text.unlikely, "xa"
..TXTST1:
# -- Begin  sgemm4_opt
	.section text.unlikely, "xa"
# mark_begin;
# Threads 2
        .align    16,0x90
	.globl sgemm4_opt
# --- sgemm4_opt(char *, char *, const int *, const int *, const int *, const float *, const float *, const int *, const float *, const int *, const float *, float *, const int *)
sgemm4_opt:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %r8
# parameter 6: %r9
# parameter 7: 16 + %rsp
# parameter 8: 24 + %rsp
# parameter 9: 32 + %rsp
# parameter 10: 40 + %rsp
# parameter 11: 48 + %rsp
# parameter 12: 56 + %rsp
# parameter 13: 64 + %rsp
..B2.1:                         # Preds ..B2.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_sgemm4_opt.55:
..L56:
                                                         #288.1
        pushq     %rsi                                          #288.1 c1
	.cfi_def_cfa_offset 16
        movq      %rdx, %rax                                    #288.1 c1
        vmovss    (%r9), %xmm0                                  #298.24 c1
        cmpb      $110, (%rdi)                                  #301.3 c1
        movq      48(%rsp), %r10                                #288.1 c3
        movq      %rsi, %rdx                                    #288.1 c3
        movslq    (%rax), %rsi                                  #291.23 c7 stall 1
        movl      (%r8), %eax                                   #292.23 c7
        vmovss    (%r10), %xmm1                                 #299.24 c11 stall 1
        jne       ..B2.16       # Prob 21%                      #301.3 c11
                                # LOE rdx rcx rbx rbp rsi r12 r13 r14 r15 eax xmm0 xmm1
..B2.2:                         # Preds ..B2.1
                                # Execution count [7.84e-01]
        cmpb      $110, (%rdx)                                  #301.3 c1
        jne       ..B2.16       # Prob 0%                       #301.3 c3
                                # LOE rcx rbx rbp rsi r12 r13 r14 r15 eax xmm0 xmm1
..B2.3:                         # Preds ..B2.2
                                # Execution count [7.80e-01]
        cmpl      $64, (%rcx)                                   #302.3 c1
        jne       ..B2.15       # Prob 5%                       #302.3 c3
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm0 xmm1
..B2.4:                         # Preds ..B2.3
                                # Execution count [7.41e-01]
        cmpq      $500, %rsi                                    #302.3 c1
        je        ..B2.6        # Prob 33%                      #302.3 c3
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm0 xmm1
..B2.5:                         # Preds ..B2.4
                                # Execution count [4.94e-01]
        cmpq      $1000, %rsi                                   #302.3 c1
        jne       ..B2.15       # Prob 50%                      #302.3 c3
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm0 xmm1
..B2.6:                         # Preds ..B2.4 ..B2.5
                                # Execution count [4.94e-01]
        cmpl      $2000, %eax                                   #302.3 c1
        jne       ..B2.15       # Prob 0%                       #302.3 c3
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm0 xmm1
..B2.7:                         # Preds ..B2.6
                                # Execution count [4.91e-01]
        vcmpss    $0, .L_2il0floatpacket.1(%rip), %xmm0, %k0    #303.3 c1
        kortestw  %k0, %k0                                      #303.3 c3
        je        ..B2.14       # Prob 21%                      #303.3 c5
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax xmm1
..B2.8:                         # Preds ..B2.7
                                # Execution count [3.85e-01]
        vxorps    %xmm0, %xmm0, %xmm0                           #303.3 c1
        vcmpss    $0, %xmm0, %xmm1, %k0                         #303.3 c3
        kortestw  %k0, %k0                                      #303.3 c5
        je        ..B2.14       # Prob 0%                       #303.3 c7
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax
..B2.9:                         # Preds ..B2.8
                                # Execution count [3.83e-01]
        movq      24(%rsp), %rdx                                #288.1 c1
        movslq    (%rdx), %rcx                                  #304.3 c5 stall 1
        cmpq      %rcx, %rsi                                    #304.3 c9 stall 1
        jne       ..B2.17       # Prob 5%                       #304.3 c11
                                # LOE rbx rbp rsi r12 r13 r14 r15 eax
..B2.10:                        # Preds ..B2.9
                                # Execution count [3.64e-01]
        movq      40(%rsp), %rdx                                #288.1 c1
        cmpl      (%rdx), %eax                                  #304.3 c5 stall 1
        jne       ..B2.17       # Prob 5%                       #304.3 c7
                                # LOE rbx rbp rsi r12 r13 r14 r15
..B2.11:                        # Preds ..B2.10
                                # Execution count [3.46e-01]
        movq      64(%rsp), %rax                                #288.1 c1
        movslq    (%rax), %rdx                                  #304.3 c5 stall 1
        cmpq      %rdx, %rsi                                    #304.3 c9 stall 1
        jne       ..B2.17       # Prob 0%                       #304.3 c11
                                # LOE rbx rbp rsi r12 r13 r14 r15
..B2.12:                        # Preds ..B2.11
                                # Execution count [3.44e-01]
        movq      32(%rsp), %rcx                                #306.3 c1
        movq      16(%rsp), %r8                                 #306.3 c1
        movl      $64, %edi                                     #306.3 c1
        movl      $2000, %edx                                   #306.3 c1
        movq      56(%rsp), %r9                                 #306.3 c5 stall 1
        addq      $8, %rsp                                      #306.3 c5
	.cfi_def_cfa_offset 8
#       sgemm_opt(const uint64_t, const uint64_t, const uint64_t, const float *__restrict__, const float *__restrict__, float *__restrict__)
        jmp       sgemm_opt                                     #306.3 c9 stall 1
	.cfi_def_cfa_offset 16
                                # LOE
..B2.14:                        # Preds ..B2.8 ..B2.7
                                # Execution count [1.08e-01]
        movl      $.L_2__STRING.4, %edi                         #303.3 c1
        movl      $.L_2__STRING.1, %esi                         #303.3 c1
        movl      $303, %edx                                    #303.3 c3
        movl      $__$U1, %ecx                                  #303.3 c3
#       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #303.3 c5
                                # LOE
..B2.15:                        # Preds ..B2.3 ..B2.5 ..B2.6
                                # Execution count [2.88e-01]
        movl      $.L_2__STRING.3, %edi                         #302.3 c1
        movl      $.L_2__STRING.1, %esi                         #302.3 c1
        movl      $302, %edx                                    #302.3 c3
        movl      $__$U1, %ecx                                  #302.3 c3
#       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #302.3 c5
                                # LOE
..B2.16:                        # Preds ..B2.1 ..B2.2
                                # Execution count [2.20e-01]
        movl      $.L_2__STRING.2, %edi                         #301.3 c1
        movl      $.L_2__STRING.1, %esi                         #301.3 c1
        movl      $301, %edx                                    #301.3 c3
        movl      $__$U1, %ecx                                  #301.3 c3
#       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #301.3 c5
                                # LOE
..B2.17:                        # Preds ..B2.11 ..B2.10 ..B2.9
                                # Execution count [3.91e-02]
        movl      $.L_2__STRING.5, %edi                         #304.3 c1
        movl      $.L_2__STRING.1, %esi                         #304.3 c1
        movl      $304, %edx                                    #304.3 c3
        movl      $__$U1, %ecx                                  #304.3 c3
#       __assert_fail(const char *, const char *, unsigned int, const char *)
        call      __assert_fail                                 #304.3 c5
        .align    16,0x90
                                # LOE
	.cfi_endproc
# mark_end;
	.type	sgemm4_opt,@function
	.size	sgemm4_opt,.-sgemm4_opt
	.data
# -- End  sgemm4_opt
	.bss
	.align 4
	.align 4
___kmpv_zerosgemm_opt_0:
	.type	___kmpv_zerosgemm_opt_0,@object
	.size	___kmpv_zerosgemm_opt_0,4
	.space 4	# pad
	.section .rodata, "a"
	.align 4
	.align 4
.L_2il0floatpacket.1:
	.long	0x3f800000
	.type	.L_2il0floatpacket.1,@object
	.size	.L_2il0floatpacket.1,4
	.align 1
__$U0:
	.long	1835362163
	.long	1886347117
	.word	116
	.type	__$U0,@object
	.size	__$U0,10
	.align 1
__$U1:
	.long	1835362163
	.long	1868510317
	.word	29808
	.byte	0
	.type	__$U1,@object
	.size	__$U1,11
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
	.align 4
.L_2__STRING.0:
	.long	723544939
	.long	1025520672
	.long	7348285
	.type	.L_2__STRING.0,@object
	.size	.L_2__STRING.0,12
	.align 4
.L_2__STRING.1:
	.long	1835885927
	.word	25390
	.byte	0
	.type	.L_2__STRING.1,@object
	.size	.L_2__STRING.1,7
	.space 1, 0x00 	# pad
	.align 4
.L_2__STRING.4:
	.long	1752198241
	.long	1027416161
	.long	808333600
	.long	640032870
	.long	1952801312
	.long	1027416161
	.long	808333344
	.word	102
	.type	.L_2__STRING.4,@object
	.size	.L_2__STRING.4,30
	.space 2, 0x00 	# pad
	.align 4
.L_2__STRING.3:
	.long	1027416173
	.long	540292640
	.long	673195558
	.long	1027416174
	.long	808465696
	.long	545029152
	.long	1027416174
	.long	808464672
	.long	639641904
	.long	544219174
	.long	840973629
	.long	3158064
	.type	.L_2__STRING.3,@object
	.size	.L_2__STRING.3,48
	.align 4
.L_2__STRING.2:
	.long	1918136362
	.long	1098083937
	.long	540884256
	.long	539455015
	.long	706749990
	.long	1634882672
	.long	541225838
	.long	656424253
	.word	10094
	.byte	0
	.type	.L_2__STRING.2,@object
	.size	.L_2__STRING.2,35
	.space 1, 0x00 	# pad
	.align 4
.L_2__STRING.5:
	.long	1299196456
	.long	540884256
	.long	1684828202
	.long	639641953
	.long	707272742
	.long	1025526640
	.long	1881808957
	.long	694314092
	.long	539371040
	.long	1299196456
	.long	540884256
	.long	1684828202
	.word	10595
	.byte	0
	.type	.L_2__STRING.5,@object
	.size	.L_2__STRING.5,51
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
# End
