
#include <math.h>
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


#define SIZE 512
#define NUM_THREADS 512
#define NUM_BLOCKS 512

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

const char *ptxCode = "                                \
   .version 7.4                                     \n \
   .target sm_35                                    \n \
   .address_size 64                                 \n \
   .visible .entry MatrixMul(                       \n \
        .param .u64 MatrixMul_param_0,              \n \
        .param .u64 MatrixMul_param_1,              \n \
        .param .u64 MatrixMul_param_2               \n \
   ){                                               \n \
        .reg .f32   %f<4>;                          \n \
        .reg .b32   %r<9>;                          \n \
        .reg .b64   %rd<14>;                        \n \
        .reg .pred  %p<2>;                          \n \
        .const .s32 SIZE = 512;                     \n \
        .shared .f32 c_cache[512];                 \n \
        ld.param.u64    %rd1, [MatrixMul_param_0];  \n \
        ld.param.u64    %rd2, [MatrixMul_param_1];  \n \
        ld.param.u64    %rd3, [MatrixMul_param_2];  \n \
        cvta.to.global.u64      %rd4, %rd1;         \n \
        cvta.to.global.u64      %rd5, %rd2;         \n \
        cvta.to.global.u64      %rd6, %rd3;         \n \
        mov.u32 %r0, %ctaid.y;  //Blockidx.y        \n \
        mov.u32 %r1, %ctaid.x;  //col = Blockidx.x  \n \
        mov.u32 %r2, %nctaid.x; //gridDim.x         \n \
        mov.u32 %r3, %tid.x;    //Threadid.x        \n \
        mov.u32 %r4, %ntid.x;   //BlockDim.x        \n \
        \n \
        \n \
        mad.lo.s32 %r5,%r0,%r2,%r3; //row + threadid.x                 \n \
        mad.lo.s32 %r6,%r3,%r4,%r1; //col + threadid.x * blockDim.x    \n \
        mul.wide.u32 %rd7, %r5, 4;     // offset * sizeof(float)    \n \
        mul.wide.u32 %rd8, %r6, 4;     // offset * sizeof(float)    \n \
        add.s64 %rd9, %rd7, %rd4;  //dev_a.m[offset]                \n \
        add.s64 %rd10, %rd8, %rd5;  //dev_b.m[offset]               \n \
        ld.global.f32 %f1, [%rd9];                                  \n \
        ld.global.f32 %f2, [%rd10];                                 \n \
        mul.f32 %f3, %f1, %f2; //dev_a * dev_b                      \n \
        \n \
        //store on c_cache                                          \n \
        mul.wide.u32 %rd11, %r3, 4; //c_cacheoffset = threadIdx     \n \
        mov.u64 %rd13, c_cache[0];                                 \n \
        add.u64 %rd12, %rd13, %rd11; //dir c_cache[threadIdx]     \n \
        st.shared.f32 [%rd12],%f3;                                 \n \
        bar.sync 0x0;                                               \n \
        \n \
        shr.u32 %r7, %r4, 1; // l = blockDim.x / 2                  \n \
        WHILE:                                                     \n \
        setp.ne.u32 %p0, %r7, 0; //CMP l != 0                       \n \
        @!%p0 bra ENDWHILE;                                         \n \
            setp.lt.u32 %p1, %r3, %r7;                              \n \
            @!%p1 bra ENDIF;     //line 50                                     \n \
                ld.shared.f32 %f1, [%rd12]; // valor c_cache        \n \
                mul.wide.u32 %rd7, %r7, 4; // l * 4                 \n \
                add.u64 %rd7, %rd7, %rd11; // threadid + l           \n \
                add.u64 %rd8, %rd13, %rd7; //dir c_cache[threadIdx]  \n \
                ld.shared.f32 %f2, [%rd8]; // valor c_cache thread+l    \n \
                add.f32 %f3, %f1, %f2; // c_cache + c_cache             \n \
                st.shared.f32 [%rd12], %f3;                             \n \
            ENDIF:                                                     \n \
            bar.sync 0x1;                                               \n \
            shr.u32 %r7, %r7, 1;                                            \n \
            bra WHILE;                                                  \n \
        ENDWHILE:                                                      \n \
        \n \
        setp.eq.u32 %p0, %r3, 0; //if threadid == 0                     \n \
        @!%p0: bra ENDIF2;                                               \n \
        ld.shared.f32 %f3, [%rd12];                                     \n \
        add.s32 %r7, %r5, %r6; //row + col                              \n \
        mul.wide.u32 %rd7, %r7, 4; //offset row+col                     \n \
        add.s64 %rd7, %rd6, %rd7;                                       \n \
        st.global.f32 [%rd7], %f3; //.param??                                     \n \
        ENDIF2:                                                         \n \
        ret;                                                            \n \
    } ";

int elfLoadAndKernelLaunch(void* elf, size_t elfSize)
{
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr dX, dY, dOut;
    size_t i;
    size_t bufferSize = SIZE * SIZE * sizeof(float);
    float a;
    float hX[SIZE*SIZE], hY[SIZE*SIZE], hOut[SIZE*SIZE];
    void* args[3];

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "MatrixMul"));

    // Generate input for execution, and create output buffers.
    for (i = 0; i < SIZE*SIZE; ++i) {
        hX[i] = (float)1;
        hY[i] = (float)1;
    }
    CUDA_SAFE_CALL(cuMemAlloc(&dX,   bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dY,   bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));

    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));

    args[0] = &dX;
    args[1] = &dY;
    args[2] = &dOut;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                   NUM_BLOCKS,  NUM_BLOCKS, 1, // grid dim
                                   NUM_THREADS, 1, 1, // block dim
                                   0, NULL, // shared mem and stream
                                   args, 0)); // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize()); // Retrieve and print output.

    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));

    float value = hOut[0];
    int count = 0;
    for (i = 0; i < SIZE*SIZE; ++i) {
         if(hOut[i] == value){
                count++;
            }else {
                printf("Result:[%ld]:%f\n", i, hOut[i]);
            }
    }
    printf("%f repited %d times\n", value, count);

    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dY));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    return 0;
}

int main(int _argc, char *_argv[])
{
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult status;

    size_t elfSize, infoSize, errorSize;
    char *elf, *infoLog, *errorLog;
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

    status = nvPTXCompilerCompile(compiler,
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

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));

    elf = (char*) malloc(elfSize);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf));

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));

    if (infoSize != 0) {
        infoLog = (char*)malloc(infoSize+1);
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));

    // Load the compiled GPU assembly code 'elf'
    elfLoadAndKernelLaunch(elf, elfSize);

    free(elf);
    return 0;
}

