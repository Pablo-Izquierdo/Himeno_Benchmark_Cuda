/*******************************************************************
HIMENO BENCHMARK 
Code writen by Pablo Izquierdo González on August, 2021.

On this code, the main functions (addgosa and main) are writen on PTX (assembly code of cuda)
to run on nvidia gpu. The objective was try to optimize more the benchmark,
trying to make the assembly code better thant de compiler (which it is quite difficult).

Note: After a long time debuging I wasn´t able to run it, due to a problem on
compilation time.

* *******************************************************************/

//#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <stdbool.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
//#include "./common/book.h"
#include "cuda.h"
#include "nvPTXCompiler.h"


#define THREADPERBLOCK_FORREDUCTION 512
#define THREADPERBLOCK_COLS 4
#define THREADPERBLOCK_DEPS 128
#define SLICE_SIZE 30

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

#define CUDA_SAFE_CALL(x)                                               \
    do {                                                                \
        CUresult result = x;                                            \
        if (result != CUDA_SUCCESS) {                                   \
            const char *msg;                                            \
            cuGetErrorName(result, &msg);                               \
            printf("error: %s failed with error %s\n", #x, msg);        \
            exit(1);                                                    \
        }                                                               \
    } while(0)

#define NVPTXCOMPILER_SAFE_CALL(x)                                       \
    do {                                                                 \
        nvPTXCompileResult result = x;                                   \
        if (result != NVPTXCOMPILE_SUCCESS) {                            \
            printf("error: %s failed with error code %d\n", #x, result); \
            exit(1);                                                     \
        }                                                                \
    } while(0)

//add_gosas
const char *ptxCode_addgosa = "                                \
   .version 7.4                                     \n \
   .target sm_35                                    \n \
   .address_size 64                                 \n \
   .visible .entry add_gosas(                       \n \
        .param .u64 add_gosas_param_0               \n \
   ){                                               \n \
        .reg .f32   %f<4>;                          \n \
        .reg .b32   %r<5>;                          \n \
        .reg .b64   %rd<5>;                         \n \
        .reg .b32   %c<1>;                          \n \
        .reg .pred  %p<2>;                          \n \
        .const .b32 THREADPERBLOCK_FORREDUCTION = 512;       \n \
        mov.b32 %c0, THREADPERBLOCK_FORREDUCTION;       \n \
        .shared .f32 c_cache[512];                             \n \
        ld.param.u64    %rd1, [add_gosas_param_0];             \n \
        cvta.to.global.u64      %rd1, %rd1; //get generic address    \n \
        mov.u32 %r0, %tid.x;    //Threadid.x                    \n \
        mov.u32 %r1, %ctaid.x;  //Blockidx.x                    \n \
        mov.u32 %r2, %ntid.x;   //BlockDim.x                    \n \
        \n \
        \n \
        mad.lo.s32 %r3,%r1,%r2,%r0; //blockid.x * blockDim.x + threadid.x    \n \
        mul.wide.u32 %rd4, %r3, 4;     // offset_dev_gosa * sizeof(float)    \n \
        add.s64 %rd4, %rd1, %rd4;  //dir dev_gosa + offset                   \n \
        ld.global.f32 %f1, [%rd4]; //dev_gosa[offset]               \n \
        \n \
        //Store on c_cache                                          \n \
        mul.wide.u32 %rd4, %r0, 4; //c_cacheoffset = threadIdx      \n \
        mov.u64 %rd2, c_cache[0];  //get c_cache dir                \n \
        add.u64 %rd3, %rd2, %rd4; //dir c_cache[threadIdx]          \n \
        st.shared.f32 [%rd3], %f1;                                  \n \
        bar.sync 0x0;                                               \n \
        \n \
        shr.u32 %r3, %r2, 1; // l = blockDim.x / 2                  \n \
        WHILE:                                                      \n \
        setp.ne.u32 %p0, %r3, 0; //CMP l != 0                       \n \
        @!%p0 bra ENDWHILE;                                         \n \
            setp.lt.u32 %p1, %r0, %r3;                              \n \
            @!%p1 bra ENDIF;                                        \n \
                ld.shared.f32 %f1, [%rd3]; // valor c_cache         \n \
                mul.wide.u32 %rd4, %r3, 4; // l * 4                       \n \
                add.u64 %rd4, %rd4, %rd3; // threadid + l              \n \
                //add.u32 %rd8, %rd2, %rd7; //dir c_cache[threadIdx];  \n \
                ld.shared.f32 %f2, [%rd3]; // valor c_cache thread+l   \n \
                add.f32 %f3, %f1, %f2; // c_cache + c_cache            \n \
                st.shared.f32 [%rd3], %f3;                             \n \
            ENDIF:                                                     \n \
            bar.sync 0x1;                                              \n \
            shr.u32 %r3, %r3, 1;  // l/=2                              \n \
            bra WHILE;                                                 \n \
        ENDWHILE:                                                      \n \
        \n \
        setp.eq.u32 %p0, %r0, 0; //if threadid == 0                     \n \
        @!%p0 bra ENDIF2;                                               \n \
        ld.shared.f32 %f3, [%rd3]; //cache[0]                           \n \
        mul.wide.u32 %rd2, %r1, 4; //offset blockid * sizeof(float)     \n \
        add.s64 %rd4, %rd1, %rd2;  // dir dev_gosa[blockid]             \n \
        st.global.f32 [%rd4], %f3; //dev_gosa[blockid] = cache[0]       \n \
        ENDIF2:                                                         \n \
        ret;                                                            \n \
    } ";


//cudaJacobi
const char *ptxCode = "                                \
   .version 7.4                                     \n \
   .target sm_35                                    \n \
   .address_size 64                                 \n \
   .visible .entry Cudajacobi(                      \n \
        .param .u64 omega_param_0,                  \n \
        .param .u64 dev_a0_param_1,                 \n \
        .param .u64 dev_a1_param_2,                 \n \
        .param .u64 dev_a2_param_3,                 \n \
        .param .u64 dev_a3_param_4,                 \n \
        .param .u64 dev_b0_param_5,                 \n \
        .param .u64 dev_b1_param_6,                 \n \
        .param .u64 dev_b2_param_7,                 \n \
        .param .u64 dev_c0_param_8,                 \n \
        .param .u64 dev_c1_param_9,                 \n \
        .param .u64 dev_c2_param_10,                \n \
        .param .u64 dev_p_param_11,                 \n \
        .param .u64 dev_bnd_param_12,               \n \
        .param .u64 dev_wrk1_param_13,              \n \
        .param .u64 dev_wrk2_param_14,              \n \
        .param .u64 dev_gosa_param_15,              \n \
        .param .u64 prows_param_16,                 \n \
        .param .u64 pcols_param_17,                 \n \
        .param .u64 pdeps_param_18                  \n \
   ){                                               \n \
        .reg .f32   %f<5>;                          \n \
        .reg .b32   %r<19>;                         \n \
        .reg .b64   %rd<25>;                         \n \
        .reg .b32   %c<2>;                             \n \
        .reg .pred  %p<3>;                             \n \
        .const .b32 THREADPERBLOCK_COLS = 4;           \n \
        .const .b32 THREADPERBLOCK_DEPS = 128;         \n \
        mov.u32 %c0, THREADPERBLOCK_COLS;              \n \
        mov.u32 %c1, THREADPERBLOCK_DEPS;              \n \
        .shared .f32 p_cache[2340];                    \n \
        mov.u64 %rd19, p_cache[0];  //p_cache[0]       \n \
        mov.u64 %rd20, p_cache[780];  //p_cache[1]     \n \
        mov.u64 %rd21, p_cache[1560];  //p_cache[2]    \n \
        ld.param.u64    %rd0, [omega_param_0];         \n \
        ld.param.u64    %rd1, [dev_a0_param_1];        \n \
        ld.param.u64    %rd2, [dev_a1_param_2];        \n \
        ld.param.u64    %rd3, [dev_a2_param_3];        \n \
        ld.param.u64    %rd4, [dev_a3_param_4];        \n \
        ld.param.u64    %rd5, [dev_b0_param_5];        \n \
        ld.param.u64    %rd6, [dev_b1_param_6];        \n \
        ld.param.u64    %rd7, [dev_b2_param_7];        \n \
        ld.param.u64    %rd8, [dev_c0_param_8];        \n \
        ld.param.u64    %rd9, [dev_c1_param_9];        \n \
        ld.param.u64    %rd10, [dev_c2_param_10];      \n \
        ld.param.u64    %rd11, [dev_p_param_11];       \n \
        ld.param.u64    %rd12, [dev_bnd_param_12];     \n \
        ld.param.u64    %rd13, [dev_wrk1_param_13];    \n \
        ld.param.u64    %rd14, [dev_wrk2_param_14];    \n \
        ld.param.u64    %rd15, [dev_gosa_param_15];         \n \
        ld.param.u64    %rd22, [prows_param_16];       \n \
        ld.param.u32    %rd23, [pcols_param_17];       \n \
        ld.param.u32    %rd24, [pdeps_param_18];       \n \
        //get generic address                          \n \
        cvta.to.global.u64      %rd0, %rd0;            \n \
        cvta.to.global.u64      %rd1, %rd1;            \n \
        cvta.to.global.u64      %rd2, %rd2;            \n \
        cvta.to.global.u64      %rd3, %rd3;            \n \
        cvta.to.global.u64      %rd4, %rd4;            \n \
        cvta.to.global.u64      %rd5, %rd5;             \n \
        cvta.to.global.u64      %rd6, %rd6;            \n \
        cvta.to.global.u64      %rd7, %rd7;            \n \
        cvta.to.global.u64      %rd8, %rd8;            \n \
        cvta.to.global.u64      %rd9, %rd9;            \n \
        cvta.to.global.u64      %rd10, %rd10;          \n \
        cvta.to.global.u64      %rd11, %rd11;          \n \
        cvta.to.global.u64      %rd12, %rd12;          \n \
        cvta.to.global.u64      %rd13, %rd13;          \n \
        cvta.to.global.u64      %rd14, %rd14;          \n \
        cvta.to.global.u64      %rd15, %rd15;           \n \
        cvta.to.global.u64      %rd22, %rd22;          \n \
        cvta.to.global.u64      %rd23, %rd23;          \n \
        cvta.to.global.u64      %rd24, %rd24;          \n \
        //common variables                             \n \
        mov.u32 %r0, %tid.x;    //Threadid.x           \n \
        mov.u32 %r1, %tid.y;    //Threadid.y           \n \
        mov.u32 %r2, %ctaid.x;  //Blockidx.x           \n \
        mov.u32 %r3, %ctaid.y;  //Blockidx.y           \n \
        ld.global.s32 %r14, [%rd22]; //dev_p.rows      \n \
        sub.s32 %r4, %r14, 2;  //imax = p.rows - 2      \n \
        ld.global.s32 %r15, [%rd23]; //dev_p.cols      \n \
        sub.s32 %r5, %r15, 1;  //jmax                  \n \
        ld.global.s32 %r16, [%rd24]; //dev_p.deps      \n \
        sub.s32 %r6, %r16, 1;  //kmax                  \n \
        mov.s32 %r1, 0; //i=0                               \n \
        mad.lo.s32 %r8, %r2, %c0, %r0; //j = blockid*COLS + threadid  \n \
        mad.lo.s32 %r9, %r3, %c1, %r1; //k = blockid*DEPS + threadid  \n \
        mad.lo.s32 %r17, %r8, %r16, %r9; //offset= j * dev_p.deps + k \n \
        //Compute                                                  \n \
        setp.gt.u32 %p2, %r0, 0; //cache_j > 0                      \n \
        setp.gt.and.u32 %p1, %r1, 0, %p2; //cache_k > 0            \n \
        setp.lt.and.u32 %p2, %r8, %r5, %p1; //j < jmax             \n \
        setp.lt.and.u32 %p1, %r9, %r6, %p2; //k < kmax             \n \
        setp.le.and.u32 %p2, %r0, %c0, %p1; //cache_j <= cols      \n \
        setp.le.and.u32 %p1, %r1, %c1, %p2; //cache_k <= deps      \n \
        \n \
        \n \
        \n \
        FOR:                                                    \n \
            setp.lt.u32 %p0, %r7, %r4; //CMP i < imax (-2)      \n \
            @!%p0 bra END;                                      \n \
            mul.lo.s32 %r10, %r7, %r15;                            \n \
            mul.lo.s32 %r10, %r10, %r16; //row                      \n \
            mad.lo.u32 %r18, %r0, %c1, %r1;                        \n \
            mul.wide.u32 %rd18, %r18, 4; // p_cache offset base \n \
            \n \
            add.u32 %r11, %r17, %r10; //offset + row                \n \
            mul.wide.u32 %rd17, %r11, 4;                            \n \
            add.u64 %rd17, %rd11, %rd17; //dir dev_p[offset+row]    \n \
            ld.global.f32 %f0, [%rd17]; //value dev_p[offset+row]   \n \
            add.u64 %rd17, %rd19, %rd18; //dir p_cache0[offsetbase] \n \
            st.shared.f32 [%rd17], %f0; // store on cache0          \n \
            \n \
            mad.lo.u32 %r11, %r15, %r16, %r11; //cols * deps + (offset + row)  \n \
            mul.wide.u32 %rd17, %r11, 4;                                    \n \
            add.u64 %rd17, %rd11, %rd17; //dir dev_p[offset+row]            \n \
            ld.global.f32 %f0, [%rd17]; //value dev_p[offset+row+cols*deps] \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]         \n \
            st.shared.f32 [%rd17], %f0; // store on cache1                  \n \
            \n \
            mad.lo.u32 %r11, %r15, %r16, %r11; //cols * deps + 2*(offset + row)    \n \
            mul.wide.u32 %rd17, %r11, 4;                                        \n \
            add.u64 %rd17, %rd11, %rd17; //dir dev_p[offset+row]                \n \
            ld.global.f32 %f0, [%rd17]; //value dev_p[offset+row+2*(cols*deps)] \n \
            add.u64 %rd17, %rd21, %rd18; //dir p_cache2[offsetbase]             \n \
            st.shared.f32 [%rd17], %f0; // store on cache2                      \n \
            bar.sync 0x0;                                                       \n \
            \n \
        //IF                                                    \n \
            @!%p1 bra END;                                     \n \
            \n \
            add.u32 %r18, %r17, %r10; //offsetbase              \n \
            mul.wide.u32 %rd16, %r18, 4; //offsetbase * 4       \n \
            mul.lo.u32 %r18, %r15, %r16; // +1                     \n \
            mul.wide.u32 %rd17, %r18, 4; // +1 * 4              \n \
            add.u64 %rd16, %rd17, %rd16; // deviceOffset base   \n \
            \n \
            //Get values and operate                                    \n \
            //dev_a                                                     \n \
            add.u64 %rd17, %rd1, %rd16; //dir dev_a0[deviceoffset]      \n \
            ld.global.f32 %f0, [%rd17]; //value dev_a0[deviceoffset]    \n \
            add.u64 %rd17, %rd21, %rd18; //dir p_cache2[offsetbase]     \n \
            ld.shared.f32 %f1, [%rd17]; //value p_cache2[offsetbase]    \n \
            add.u64 %rd17, %rd2, %rd16; //dir dev_a1[deviceoffset]      \n \
            ld.global.f32 %f2, [%rd17]; //value dev_a1[deviceoffset]    \n \
            mul.wide.s32 %rd22, %c1, 4; // +1 (cache_j)                    \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]                         \n \
            add.u64 %rd17, %rd17, %rd22; //dir p_cache1[offsetbase+128]             \n \
            ld.shared.f32 %f3, [%rd17]; //value p_cache1[offsetbase + 128]          \n \
            mul.f32 %f2, %f2, %f3; // dev_a1 * p_cache1                             \n \
            mad.rn.f32 %f0, %f0, %f1, %f2; //dev_a0 *p_cache2 + (dev_a1 * p_cache1) \n \
            add.u64 %rd17, %rd3, %rd16; //dir dev_a2[deviceoffset]                  \n \
            ld.global.f32 %f1, [%rd17]; //value dev_a2[deviceoffset]                \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]                 \n \
            ld.shared.f32 %f2, [%rd17 + 4]; //value p_cache1[offsetbase + 1]        \n \
            mad.rn.f32 %f0, %f1, %f2, %f0; // dev_a2 * p_cache1 + s0                \n \
            \n \
            //dev_b0                                                                \n \
            add.u64 %rd17, %rd5, %rd16; //dir dev_b0[deviceoffset]                  \n \
            ld.global.f32 %f1, [%rd17]; //value dev_b0[deviceoffset]                \n \
            add.u64 %rd17, %rd21, %rd18; //dir p_cache2[offsetbase]                 \n \
            add.u64 %rd23, %rd17, %rd22; //dir p_cache2[offsetbase + 128]           \n \
            ld.shared.f32 %f2, [%rd23]; //value p_cache2[offsetbase + 128]          \n \
            sub.u64 %rd23, %rd17, %rd22; //dir p_cache2[offsetbase - 128]           \n \
            ld.shared.f32 %f3, [%rd23]; //value p_cache2[offsetbase - 128]          \n \
            sub.f32 %f2, %f2, %f3; //p_cache2[base + 128] - p_cache2[base - 128]    \n \
            add.u64 %rd17, %rd19, %rd18; //dir p_cache0[offsetbase]                 \n \
            add.u64 %rd23, %rd17, %rd22; //dir p_cache0[offsetbase + 128]           \n \
            ld.shared.f32 %f3, [%rd23]; //value p_cache0[offsetbase + 128]          \n \
            sub.f32 %f2, %f2, %f3; // old sub - p_cache0[offsetbase + 128]          \n \
            sub.u64 %rd23, %rd17, %rd22; //dir p_cache0[offsetbase - 128]           \n \
            ld.shared.f32 %f3, [%rd23]; //value p_cache0[offsetbase - 128]          \n \
            add.f32 %f2, %f2, %f3; // old sub + p_cache0[offsetbase - 128]          \n \
            mad.rn.f32 %f0, %f1, %f2, %f0; // dev_b0 * (p_cache subs) + S0          \n \
            \n \
            //dev_b1                                                                \n \
            add.u64 %rd17, %rd6, %rd16; //dir dev_b1[deviceoffset]                  \n \
            ld.global.f32 %f1, [%rd17]; //value dev_b1[deviceoffset]                \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]                 \n \
            add.u64 %rd23, %rd22, 4; // 128 +1 = 129                                \n \
            add.u64 %rd24, %rd17, %rd22; //dir p_cache1[offsetbase + 129            \n \
            ld.shared.f32 %f2, [%rd24]; //value p_cache1[offsetbase + 129]          \n \
            sub.u64 %rd24, %rd22, 4; // 128 -1 = 127                                \n \
            sub.u64 %rd23, %rd17, %rd24; //dir p_cache1[offsetbase - 127]           \n \
            ld.shared.f32 %f3, [%rd23]; //value p_cache1[offsetbase - 127]          \n \
            sub.f32 %f2, %f2, %f3; //p_cache1[base + 129] - p_cache1[base - 127]    \n \
            add.u64 %rd23, %rd17, %rd24; //dir p_cache1[offsetbase + 127]           \n \
            ld.shared.f32 %f3, [%rd23]; //value p_cache1[offsetbase + 127]          \n \
            sub.f32 %f2, %f2, %f3; // old sub - p_cache1[offsetbase + 127]          \n \
            add.u64 %rd23, %rd22, 4; // 128 +1 = 129                                \n \
            sub.u64 %rd23, %rd17, %rd23; // dir p_cache1[offsetbase - 129]          \n \
            ld.shared.f32 %f3, [%rd23]; //value p_cache1[offsetbase - 129]          \n \
            add.f32 %f2, %f2, %f3; // old sub + p_cache1[offsetbase - 129]          \n \
            mad.rn.f32 %f0, %f1, %f2, %f0; // dev_b1 * (p_cache subs) + S0          \n \
            \n \
            //dev_b2                                                                \n \
            add.u64 %rd17, %rd7, %rd16; //dir dev_b2[deviceoffset]                          \n \
            ld.global.f32 %f1, [%rd17]; //value dev_b2[deviceoffset]                \n \
            add.u64 %rd17, %rd21, %rd18; //dir p_cache2[offsetbase]                 \n \
            ld.shared.f32 %f2, [%rd17 + 4]; //value p_cache2[offsetbase + 4]        \n \
            ld.shared.f32 %f4, [%rd17 + -4]; //value p_cache2[offsetbase - 4]       \n \
            add.u64 %rd17, %rd19, %rd18; //dir p_cache0[offsetbase]                 \n \
            ld.shared.f32 %f3, [%rd17 + 4]; //value p_cache0[offsetbase + 4]        \n \
            sub.f32 %f2, %f2, %f3; //p_cache2[base + 4] - p_cache0[base + 4]        \n \
            sub.f32 %f2, %f2, %f4; // old sub - p_cache2[offsetbase - 4]            \n \
            ld.shared.f32 %f3, [%rd17 + -4]; //value p_cache0[offsetbase - 4]       \n \
            add.f32 %f2, %f2, %f3; // old sub + p_cache0[offsetbase - 4]            \n \
            mad.rn.f32 %f0, %f1, %f2, %f0; // dev_b2 * (p_cache subs) + S0          \n \
            \n \
            //dev_c0                                                                \n \
            add.u64 %rd17, %rd8, %rd16; //dir dev_c0[deviceoffset]                  \n \
            ld.global.f32 %f1, [%rd17]; //value dev_c0[deviceoffset]                \n \
            add.u64 %rd17, %rd19, %rd18; //dir p_cache0[offsetbase]                 \n \
            ld.shared.f32 %f2, [%rd17]; //value p_cache0[offsetbase]                \n \
            mad.rn.f32 %f0, %f1, %f2, %f0; //dev_c0 * p_cache0 + S0                 \n \
            \n \
            //dev_c1                                                                \n \
            add.u64 %rd17, %rd9, %rd16; //dir dev_c1[deviceoffset]                  \n \
            ld.global.f32 %f1, [%rd17]; //value dev_c1[deviceoffset]                \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]                 \n \
            sub.u64 %rd17, %rd17, %rd22;  //dir p_cache1[offsetbase - 128]          \n \
            ld.shared.f32 %f2, [%rd17]; //value p_cache1[offsetbase - 128]          \n \
            mad.rn.f32 %f0, %f1, %f2, %f0; //dev_c1 * p_cache1 + S0                 \n \
            \n \
            //dev_c2                                                                \n \
            add.u64 %rd17, %rd10, %rd16; //dir dev_c2[deviceoffset]                 \n \
            ld.global.f32 %f1, [%rd17]; //value dev_c2[deviceoffset]                \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]                 \n \
            ld.shared.f32 %f2, [%rd17 + -4]; //value p_cache1[offsetbase - 4]       \n \
            mad.rn.f32 %f0, %f1, %f2, %f0; //dev_c1 * p_cache1 + S0                 \n \
            \n \
            //dev_wrk1                                                              \n \
            add.u64 %rd17, %rd13, %rd16; //dir dev_wrk1[deviceoffset]               \n \
            ld.global.f32 %f1, [%rd17]; //value dev_wrk1[deviceoffset]              \n \
            add.f32 %f0, %f1, %f0; //dev_c1 + S0                                    \n \
            //S0 got it on f0                                                       \n \
            \n \
            \n \
            //calculate ss                                                          \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]                 \n \
            ld.shared.f32 %f2, [%rd17]; //value p_cache1[offsetbase]                \n \
            add.u64 %rd17, %rd12, %rd16; //dir dev_bnd[deviceoffset]                \n \
            ld.global.f32 %f3, [%rd17]; //value dev_bnd[deviceoffset]               \n \
            mul.f32 %f2, %f2, %f3; // p_cache1 * dev_bnd                            \n \
            add.u64 %rd17, %rd4, %rd16; //dir dev_a3[deviceoffset]                  \n \
            ld.global.f32 %f1, [%rd17]; //value dev_a3[deviceoffset]                \n \
            mul.f32 %f1, %f0, %f1; // s0 * dev_a3                                           \n \
            sub.f32 %f1, %f1, %f2; // ss                                            \n \
            \n \
            //save value in dev_gosa                                                \n \
            add.u32 %r11, %r17, %r10;                                               \n \
            mul.wide.u32 %rd17, %r11, 4;  // offset + row                           \n \
            add.u64 %rd17, %rd17, %rd15;  // dir dev_gosa[offset+row]               \n \
            ld.global.f32 %f2, [%rd17]; //value dev_gosa[offset+row]                \n \
            mad.rn.f32 %f2, %f1, %f1, %f2; // (ss * ss) + dev_gosa                  \n \
            st.global.f32 [%rd17], %f2; // store on dev_gosa                        \n \
            \n \
            //dev_wrk2                                                              \n \
            add.u64 %rd17, %rd20, %rd18; //dir p_cache1[offsetbase]                 \n \
            ld.shared.f32 %f2, [%rd17]; //value p_cache1[offsetbase]                \n \
            ld.global.f32 %f3, [%rd0]; //value omega                                \n \
            add.u64 %rd17, %rd14, %rd16; //dir dev_wrk2[deviceoffset]               \n \
            mad.rn.f32 %f2, %f1, %f3, %f2; // omega * ss + p_cache                     \n \
            st.global.f32 [%rd17], %f2; // store on dev_wrk2                        \n \
            \n \
            bra FOR;                                                                \n \
        \n \
        END:                                                    \n \
        ret;                                                    \n \
    } ";

double fflop(int mx, int my, int mz)
{
    return((double)(mz - 2) * (double)(my - 2) * (double)(mx - 2) * 34.0);
}

double mflops(int nn, double cpu, double flop)
{
    return(flop / cpu * 1.e-6 * (double)nn);
}

/** second
    * This fuction return the time on the instant that it is call
    * */
double second()
{

    struct timeval tm;
    double t;

    static int base_sec = 0, base_usec = 0;

    gettimeofday(&tm, NULL);

    if (base_sec == 0 && base_usec == 0)
    {
        base_sec = tm.tv_sec;
        base_usec = tm.tv_usec;
        t = 0.0;
    }
    else {
        t = (double)(tm.tv_sec - base_sec) +
            ((double)(tm.tv_usec - base_usec)) / 1.0e6;
    }

    return t;
}

void set_param(int is[], char* size)
{
    if (!strcmp(size, "XS") || !strcmp(size, "xs")) {
        is[0] = 32;
        is[1] = 32;
        is[2] = 64;
        return;
    }
    if (!strcmp(size, "S") || !strcmp(size, "s")) {
        is[0] = 64;
        is[1] = 64;
        is[2] = 128;
        return;
    }
    if (!strcmp(size, "M") || !strcmp(size, "m")) {
        is[0] = 128;
        is[1] = 128;
        is[2] = 256;
        return;
    }
    if (!strcmp(size, "L") || !strcmp(size, "l")) {
        is[0] = 256;
        is[1] = 256;
        is[2] = 512;
        return;
    }
    if (!strcmp(size, "XL") || !strcmp(size, "xl")) {
        is[0] = 512;
        is[1] = 512;
        is[2] = 1024;
        return;
    }
    else {
        printf("Invalid input character !!\n");
        exit(6);
    }
}

void transferData(CUdeviceptr d, float* h, int n, int sliceStart, int sliceEnd, int rows, int cols, int deps)
{
    rows = (sliceEnd - sliceStart + 1);

    size_t rect = cols * deps;
    size_t transfer_size = rows * rect * sizeof(float); // dev->mrows = 32
    size_t offset = (n * rows + sliceStart) * rect;
    float* hst_pos = h + offset;
 
    CUDA_SAFE_CALL(cuMemcpyHtoD(d, hst_pos, transfer_size));

}

void returnData(CUdeviceptr d, float* h, int n, int sliceStart, int sliceEnd, int rows, int cols, int deps)
{
    size_t rect = cols * deps;
    size_t transfer_size = rows * rect * sizeof(float); // dev->mrows = 32
    size_t offset = (n * rows + sliceStart) * rect;
    float* hst_pos = h + offset;

    CUDA_SAFE_CALL(cuMemcpyDtoH(hst_pos + rect, d + rect,
        transfer_size - rect * sizeof(float)));
        
}

float elfLoadAndKernelLaunch(void* elf, void* elf_addgosa, size_t elfSize, int size[], int nn)
{
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUmodule module_addgosa;
    CUfunction kernel;
    CUfunction kernel_addgosa;
    CUdeviceptr dp, dbnd, dwrk1, dwrk2, da0, da1, da2, da3, db0, db1, db2,
                dc0, dc1, dc2, dgosa;

    int mimax, mjmax, mkmax, box, imax, jmax, kmax;
    mimax = size[0];
    mjmax = size[1];
    mkmax = size[2];
    imax = mimax - 1;
    jmax = mjmax - 1;
    kmax = mkmax - 1;
    size_t i;
    size_t boxSize = mimax * mjmax * mkmax * sizeof(float);
    size_t SliceSize = 32 * mjmax * mkmax * sizeof(float);
    float a, gosa;
    float *hp, *hbnd, *hwrk1, *hwrk2, *ha0, *ha1, *ha2, *ha3, *hb0, *hb1, *hb2, *hc0, *hc1, *hc2, *hgosa;
        
    hp = (float*)malloc(boxSize);
    hbnd = (float*)malloc(boxSize);
    hwrk1 = (float*)malloc(boxSize);
    hwrk2 = (float*)malloc(boxSize);
    ha0 = (float*)malloc(boxSize);
    ha1 = (float*)malloc(boxSize);
    ha2 = (float*)malloc(boxSize);
    ha3 = (float*)malloc(boxSize);
    hb0 = (float*)malloc(boxSize);
    hb1 = (float*)malloc(boxSize);
    hb2 = (float*)malloc(boxSize);
    hc0 = (float*)malloc(boxSize);
    hc1 = (float*)malloc(boxSize);
    hc2 = (float*)malloc(boxSize);
    hgosa = (float*)malloc((box / THREADPERBLOCK_FORREDUCTION) * sizeof(float));
    float omega = 0.8;
    void* args[19];
    void* args_addgosa[1];

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module_addgosa, elf_addgosa, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "Cudajacobi"));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel_addgosa, module_addgosa, "add_gosas"));

    // Generate input for execution, and create output buffers.
    for (i = 0; i < (mimax*mjmax*mkmax); ++i) {
        hp[i] = (float)(i * i) / (float)((mimax - 1) * (mimax - 1));
        hbnd[i] = (float)1;
        hwrk1[i] = (float)0;
        hwrk2[i] = (float)(i * i) / (float)((mimax - 1) * (mimax - 1));
        ha0[i] = (float)1;
        ha1[i] = (float)1;
        ha2[i] = (float)1;
        ha3[i] = (float)(1.0 / 6.0);
        hb0[i] = (float)0;
        hb1[i] = (float)0;
        hb2[i] = (float)0;
        hc0[i] = (float)1;
        hc1[i] = (float)1;
        hc2[i] = (float)1;
        hgosa[i] = (float)0;
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dp,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dbnd,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dwrk1, SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dwrk2,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&da0,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&da1, SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&da2,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&da3,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&db0, SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&db1,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&db2,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dc0, SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dc1,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dc2,   SliceSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dgosa,   boxSize));
    
    int blocksForReduction = (box / THREADPERBLOCK_FORREDUCTION);
    int threadsPerBlockForReduction = (THREADPERBLOCK_FORREDUCTION);
    float* swap = (float*)malloc(boxSize);

    int threadsPerBlockX = (THREADPERBLOCK_COLS + 2);
    int threadsPerBlockY = (THREADPERBLOCK_DEPS + 2);
    int blocksX = ((jmax + THREADPERBLOCK_COLS - 1) / THREADPERBLOCK_COLS);
    int blocksY = ((kmax + THREADPERBLOCK_DEPS - 1)/ THREADPERBLOCK_DEPS);
    printf(" Blocks: {%d, %d, %d} threads. Grid: {%d, %d, %d} blocks.\n", threadsPerBlockX, threadsPerBlockY, 1, blocksX, blocksY, 1);

    for (int n = 0; n < nn; n++) {
        for (i = 0; i < (mimax*mjmax*mkmax); ++i) {
            hgosa[i] = (float)0;
        }
        gosa = 0.f;

        //execute different slices
        // iter 0 -> send 0 to 31
        // iter 1 -> send 30 to 61
        // iter 2 -> send 60 to 91
        // iter 3 -> send 90 to ....
        for (int slice = 0; slice < mimax - 2; slice += SLICE_SIZE) { // desde 0 hasta 29 con paso 30
            int end = min((slice + SLICE_SIZE + 1), imax); // desde 0 a 31 o el final

            //Copy Struct to de GPU like linear array
            transferData(dp, hp, 1, slice, end, mimax, mjmax, mkmax);
            transferData(dbnd, hbnd, 1, slice, end, mimax, mjmax, mkmax);
            transferData(dwrk1, hwrk1, 1, slice, end, mimax, mjmax, mkmax);
            transferData(dwrk2, hwrk2, 1, slice, end, mimax, mjmax, mkmax);
            transferData(da0, ha0, 1, slice, end, mimax, mjmax, mkmax);
            transferData(da1, ha1, 1, slice, end, mimax, mjmax, mkmax);
            transferData(da2, ha2, 1, slice, end, mimax, mjmax, mkmax);
            transferData(da3, ha3, 1, slice, end, mimax, mjmax, mkmax);
            transferData(db0, hb0, 1, slice, end, mimax, mjmax, mkmax);
            transferData(db1, hb1, 1, slice, end, mimax, mjmax, mkmax);
            transferData(db2, hb2, 1, slice, end, mimax, mjmax, mkmax);
            transferData(dc0, hc0, 1, slice, end, mimax, mjmax, mkmax);
            transferData(dc1, hc1, 1, slice, end, mimax, mjmax, mkmax);
            transferData(dc2, hc2, 1, slice, end, mimax, mjmax, mkmax);
            transferData(dgosa, hgosa, 1, slice, end, mimax, mjmax, mkmax);

            /*
            CUDA_SAFE_CALL(cuMemcpyHtoD(dp, hp, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(dbnd, hbnd, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(dwrk1, hwrk1, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(dwrk2, hwrk2, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(da0, ha0, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(da1, ha1, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(da2, ha2, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(da3, ha3, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(db0, hb0, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(db1, hb1, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(db2, hb2, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(dc0, hc0, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(dc1, hc1, bufferSize));
            CUDA_SAFE_CALL(cuMemcpyHtoD(dc2, hc2, bufferSize));*/

            args[0] = &omega;
            args[1] = &da0;
            args[2] = &da1;
            args[3] = &da2;
            args[4] = &da3;
            args[5] = &db0;
            args[6] = &db1;
            args[7] = &db2;
            args[8] = &dc0;
            args[9] = &dc1;
            args[10] = &dc2;
            args[11] = &dp;
            args[12] = &dbnd;
            args[13] = &dwrk1;
            args[14] = &dwrk2;
            args[15] = &dgosa;
            args[16] = &mimax;
            args[17] = &mjmax;
            args[18] = &mkmax;

            CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                   blocksX,  blocksY, 1, // grid dim
                                   threadsPerBlockX, threadsPerBlockY, 1, // block dim
                                   0, NULL, // shared mem and stream
                                   args, 0)); // arguments
            CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.

            returnData(dwrk2, hwrk2, 1, slice, end, mimax, mjmax, mkmax);
        }

        swap = hp;
        hp = hwrk2;
        hwrk2 = swap;

        args_addgosa[0] = &dgosa;

        CUDA_SAFE_CALL( cuLaunchKernel(kernel_addgosa,
                                   blocksForReduction,  1, 1, // grid dim
                                   threadsPerBlockForReduction, 1, 1, // block dim
                                   0, NULL, // shared mem and stream
                                   args_addgosa, 0)); // arguments
        CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.

        CUDA_SAFE_CALL(cuMemcpyDtoH(hgosa, dgosa, boxSize));

        for (int z = 0; z < (box / THREADPERBLOCK_FORREDUCTION); z++) {
            gosa += hgosa[z];
        }

    } 


    float value = hgosa[0];
    int count = 0;
    for (i = 0; i < (mimax*mjmax*mkmax); ++i) {
         if(hgosa[i] == value){
                count++;
            }else {
                printf("Result:[%ld]:%f\n", i, hgosa[i]);
            }
    }
    printf("%f repited %d times\n", value, count);

    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(da0));
    CUDA_SAFE_CALL(cuMemFree(da1));
    CUDA_SAFE_CALL(cuMemFree(da2));
    CUDA_SAFE_CALL(cuMemFree(da3));
    CUDA_SAFE_CALL(cuMemFree(db0));
    CUDA_SAFE_CALL(cuMemFree(db1));
    CUDA_SAFE_CALL(cuMemFree(db2));
    CUDA_SAFE_CALL(cuMemFree(dc0));
    CUDA_SAFE_CALL(cuMemFree(dc1));
    CUDA_SAFE_CALL(cuMemFree(dc2));
    CUDA_SAFE_CALL(cuMemFree(dp));
    CUDA_SAFE_CALL(cuMemFree(dbnd));
    CUDA_SAFE_CALL(cuMemFree(dwrk1));
    CUDA_SAFE_CALL(cuMemFree(dwrk2));
    CUDA_SAFE_CALL(cuMemFree(dgosa));
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuModuleUnload(module_addgosa));
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    return gosa;
}

int main(int _argc, char *_argv[])
{
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompilerHandle compiler_addgosa = NULL;
    nvPTXCompileResult status;
    nvPTXCompileResult status_addgosa;

    size_t elfSize, elfSize_addgosa, infoSize, infoSize_addgosa, errorSize;
    char *elf, *infoLog, *errorLog, *elf_addgosa;
    unsigned int minorVer, majorVer;

    const char* compile_options[] = { "--gpu-name=sm_35",
                                      "--verbose"
                                    };

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
    printf("Current PTX Compiler API Version : %d.%d\n", majorVer, minorVer);

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler,
                                                (size_t)strlen(ptxCode),  /* ptxCodeLen */
                                                ptxCode)                  /* ptxCode */
                            );
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler_addgosa,
                                                (size_t)strlen(ptxCode_addgosa),  /* ptxCodeLen */
                                                ptxCode_addgosa)                  /* ptxCode */
                            );                        

    status = nvPTXCompilerCompile(compiler,
                                  2,                 /* numCompileOptions */
                                  compile_options);  /* compileOptions */

    status_addgosa = nvPTXCompilerCompile(compiler_addgosa,
                                  2,                 /* numCompileOptions */
                                  compile_options);  /* compileOptions */


    if (status != NVPTXCOMPILE_SUCCESS) {
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));

        if (errorSize != 0) {
            errorLog = (char*)malloc(errorSize+1);
            NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(compiler, errorLog));
            printf("Error log: %s\n", errorLog);
            free(errorLog);
        }
        exit(1);
    }

    if (status_addgosa != NVPTXCOMPILE_SUCCESS) {
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler_addgosa, &errorSize));

        if (errorSize != 0) {
            errorLog = (char*)malloc(errorSize+1);
            NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(compiler_addgosa, errorLog));
            printf("Error log addgosa: %s\n", errorLog);
            free(errorLog);
        }
        exit(1);
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler_addgosa, &elfSize_addgosa));

    elf = (char*) malloc(elfSize);
    elf_addgosa = (char*) malloc(elfSize);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf));
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler_addgosa, (void*)elf_addgosa));

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler_addgosa, &infoSize_addgosa));

    if (infoSize != 0) {
        infoLog = (char*)malloc(infoSize+1);
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }

    if (infoSize_addgosa != 0) {
        infoLog = (char*)malloc(infoSize_addgosa+1);
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler_addgosa, infoLog));
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler_addgosa));

    //MY MAIN

    int    msize[3];
    char   size[10];
    double  cpu0, cpu1, cpu, flop;
    float gosa;

    if (_argc == 2) {
        strcpy(size, _argv[1]); //1 parameter XS, S, M, L, XL
    }
    else {
        printf("For example: \n");
        printf(" Grid-size= XS (32x32x64)\n");
        printf("\t    S  (64x64x128)\n");
        printf("\t    M  (128x128x256)\n");
        printf("\t    L  (256x256x512)\n");
        printf("\t    XL (512x512x1024)\n\n");
        printf("Grid-size = ");
        scanf("%s", size);
        printf("\n");
    }

    set_param(msize, size); //set the parameters [mimax][mjmax][mkmax] on msize depending on the main parameter

    // Load the compiled GPU assembly code 'elf'
    elfLoadAndKernelLaunch(elf, elf_addgosa, elfSize, msize, 1);

    printf(" Start rehearsal measurement process.\n");
    printf(" Measure the performance in %d times.\n\n", 3);

    cpu0 = second(); //get initial time
    gosa = elfLoadAndKernelLaunch(elf, elf_addgosa, elfSize, msize, 3);
    cpu1 = second(); //get final time
    cpu = cpu1 - cpu0;
    flop = fflop(msize[0]-1, msize[1]-1, msize[2]-1);

    printf(" MFLOPS: %f time(s): %f %e\n\n",
        mflops(3, cpu, flop), cpu, gosa);
    

    free(elf);
    return 0;
}
