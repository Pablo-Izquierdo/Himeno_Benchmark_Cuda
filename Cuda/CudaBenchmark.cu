/*******************************************************************

This benchmark test program is measuring a cpu performance
of floating point operation by a Poisson equation solver.

If you have any question, please ask me via email.
written by Ryutaro HIMENO, November 26, 2001.
Version 3.0
----------------------------------------------
Ryutaro Himeno, Dr.of Eng.
Head of Computer Information Division,
RIKEN(The Institute of Pysicaland Chemical Research)
Email : himeno@postman.riken.go.jp
-------------------------------------------------------------- -
You can adjust the size of this benchmark code to fit your target
computer.In that case, please chose following sets of
[mimax][mjmax][mkmax]:
small: 33, 33, 65
small : 65, 65, 129
midium : 129, 129, 257
large : 257, 257, 513
ext.large : 513, 513, 1025
This program is to measure a computer performance in MFLOPS
by using a kernel which appears in a linear solver of pressure
Poisson eq.which appears in an incompressible Navier - Stokes solver.
A point - Jacobi method is employed in this solver as this method can
be easyly vectrized and be parallelized.
------------------
Finite - difference method, curvilinear coodinate system
Vectorizableand parallelizable on each grid point
No.of grid points : imax x jmax x kmax including boundaries
------------------
A, B, C : coefficient matrix, wrk1 : source term of Poisson equation
wrk2 : working area, OMEGA : relaxation parameter
BND : control variable for boundariesand objects(= 0 or 1)
P : pressure
* *******************************************************************/

//C Includes
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <iostream>

//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "./common/book.h"

/*define preprocessor macro
* This let us write MR(mt,n,r,c,d) insted of mt->m[(n) * mt->mrows * mt->mcols * mt->mdeps + (r) * mt->mcols* mt->mdeps + (c) * mt->mdeps + (d)]
*
* This expresion is use to select the 3D matrix index and fill it. So the mt,n,r,c,and d are variables that are reemplaced in the expresion
*
* */
#define MR(mt,n,r,c,d)  mt->m[(n) * mt->mrows * mt->mcols * mt->mdeps + (r) * mt->mcols* mt->mdeps + (c) * mt->mdeps + (d)]

/*My own macro to use on the CUDA funtion*/
#define CMR(mt,r,c,d)  mt.m[(r) * mt.mcols * mt.mdeps + (c) * mt.mdeps + (d)]


struct Mat {
    float* m;
    int mnums;
    int mrows;
    int mcols;
    int mdeps;
};

/* prototypes */
typedef struct Mat Matrix;

int newMat(Matrix* Mat, int mnums, int mrows, int mcols, int mdeps);
void clearMat(Matrix* Mat);
void set_param(int i[], char* size);
void mat_set(Matrix* Mat, int l, float z);
void mat_set_init(Matrix* Mat);
__global__  void Cudajacobi(float omega, Matrix dev_a0, Matrix dev_a1, Matrix dev_a2, Matrix dev_a3,
    Matrix dev_b0, Matrix dev_b1, Matrix dev_b2, Matrix dev_c0, Matrix dev_c1, Matrix dev_c2, Matrix dev_p,
    Matrix dev_bnd, Matrix dev_wrk1, Matrix dev_wrk2, float* dev_gosa);
float jacobi(int nn, Matrix a, Matrix b, Matrix c, Matrix p, Matrix bnd, Matrix wrk1, Matrix wrk2);
int createDevMat(Matrix* dev, int rows, int cols, int deps);
void transferData(Matrix* dest, Matrix* src, int n, int sliceStart, int sliceEnd, cudaMemcpyKind k);
__global__ void add_gosas(float* dev_gosa, float* dev_gosas_reducted);
double fflop(int, int, int);
double mflops(int, double, double);
double second();

//global variables
float   omega = 0.8;
Matrix  a, b, c, p, bnd, wrk1, wrk2;
//Matrix dev_a0, dev_a1, dev_a2, dev_a3, dev_b0, dev_b1, dev_b2,
//    dev_c0, dev_c1, dev_c2, dev_p, dev_bnd, dev_wrk1, dev_wrk2;

int main(int argc, char* argv[])
{

    int nn;
    int    imax, jmax, kmax, mimax, mjmax, mkmax, msize[3];
    float  target, gosa;
    double  cpu0, cpu1, cpu, flop;
    char   size[10];

    if (argc == 2) {
        strcpy(size, argv[1]); //1 parameter XS, S, M, L, XL
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

    nn = 3;
    mimax = msize[0];
    mjmax = msize[1];
    mkmax = msize[2];
    imax = mimax - 1;
    jmax = mjmax - 1;
    kmax = mkmax - 1;

    target = 60.0;

    printf("mimax = %d mjmax = %d mkmax = %d\n", mimax, mjmax, mkmax);
    printf("imax = %d jmax = %d kmax =%d\n", imax, jmax, kmax);

    /*
    *    Initializing 3D float matrixes
    */
    //newMat now save space for the arrat on the device
    printf("Generating matrixes and saving space on device\n");
    newMat(&p, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&bnd, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&wrk1, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&wrk2, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&a, 4, mimax, mjmax, mkmax); // 4 matrixes inicialize with 0.0
    newMat(&b, 3, mimax, mjmax, mkmax); // 3 matrixes inicialize with 0.0
    newMat(&c, 3, mimax, mjmax, mkmax); // 3 matrixes inicialize with 0.0

    mat_set_init(&p); //put MR of p inicialize the matrix of the structure with (i*i)/(mrows-1*mrows-1)
    mat_set(&bnd, 0, 1.0); //put MR of bnd matrix 0 to 1.0
    mat_set(&wrk1, 0, 0.0); //put MR of wrk1 matrix 0 to 0.0
    mat_set_init(&wrk2);
    //mat_set(&wrk2, 0, 0.0); //The original code make this but with it, it doesn´t work properlly
    mat_set(&a, 0, 1.0); //put MR of a matrix 0 to 1.0
    mat_set(&a, 1, 1.0); //put MR of a matrix 1 to 1.0
    mat_set(&a, 2, 1.0); //put MR of a matrix 2 to 1.0
    mat_set(&a, 3, 1.0 / 6.0); //put MR of a matrix 3 to 0.16
    mat_set(&b, 0, 0.0); //put MR of b matrix 0 to 0.0
    mat_set(&b, 1, 0.0); //put MR of b matrix 1 to 0.0
    mat_set(&b, 2, 0.0); //put MR of b matrix 2 to 0.0
    mat_set(&c, 0, 1.0); //put MR of c matrix 0 to 1.0
    mat_set(&c, 1, 1.0); //put MR of c matrix 1 to 1.0
    mat_set(&c, 2, 1.0); //put MR of c matrix 2 to 1.0



    /*
    *    Start measuring
    */
    gosa = jacobi(1, a, b, c, p, bnd, wrk1, wrk2);

    nn = 3;
    printf(" Start rehearsal measurement process.\n");
    printf(" Measure the performance in %d times.\n\n", nn);

    cpu0 = second(); //get initial time
    gosa = jacobi(nn, a, b, c, p, bnd, wrk1, wrk2); //use cache bandwith
    cpu1 = second(); //get final time
    cpu = cpu1 - cpu0;
    flop = fflop(imax, jmax, kmax);

    printf(" MFLOPS: %f time(s): %f %e\n\n",
        mflops(nn, cpu, flop), cpu, gosa);

    nn = (int)(target / (cpu / 3.0));

    printf(" Now, start the actual measurement process.\n");
    printf(" The loop will be excuted in %d times\n", nn);
    printf(" This will take about one minute.\n");
    printf(" Wait for a while\n\n");

    cpu0 = second();
    gosa = jacobi(nn, a, b, c, p, bnd, wrk1, wrk2);
    cpu1 = second();
    cpu = cpu1 - cpu0;

    printf(" Loop executed for %d times\n", nn);
    printf(" Gosa : %e \n", gosa);
    printf(" MFLOPS measured : %f\tcpu : %f\n", mflops(nn, cpu, flop), cpu);
    printf(" Score based on Pentium III 600MHz using Fortran 77: %f\n",
        mflops(nn, cpu, flop) / 82.84);

    /*
    *   Matrix free
    */
    clearMat(&p);
    clearMat(&bnd);
    clearMat(&wrk1);
    clearMat(&wrk2);
    clearMat(&a);
    clearMat(&b);
    clearMat(&c);

    return (0);
}

double fflop(int mx, int my, int mz)
{
    return((double)(mz - 2) * (double)(my - 2) * (double)(mx - 2) * 34.0);
}

double mflops(int nn, double cpu, double flop)
{
    return(flop / cpu * 1.e-6 * (double)nn);
}

/** set_param
* Funtion that recognice the parameter introduced and set the params in the array is[]
* */
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

/** newMat
* Inicialize the structure Mat with parameters and malloc of the matrix (each element = 0.0)
* */
int newMat(Matrix* Mat, int mnums, int mrows, int mcols, int mdeps)
{
    Mat->mnums = mnums;
    Mat->mrows = mrows;
    Mat->mcols = mcols;
    Mat->mdeps = mdeps;
    Mat->m = NULL;
    Mat->m = (float*)
        malloc(mnums * mrows * mcols * mdeps * sizeof(float));


    return(Mat->m != NULL) ? 1 : 0;
}

void clearMat(Matrix* Mat)
{
    if (Mat->m) {
        free(Mat->m);
    }
    Mat->m = NULL;
    Mat->mnums = 0;
    Mat->mcols = 0;
    Mat->mrows = 0;
    Mat->mdeps = 0;
}

/** mat_set
* Inicialize the matrix (l) of the structure with val
* */
void mat_set(Matrix* Mat, int l, float val)
{
    int i, j, k;

    for (i = 0; i < Mat->mrows; i++)
        for (j = 0; j < Mat->mcols; j++)
            for (k = 0; k < Mat->mdeps; k++)
                MR(Mat, l, i, j, k) = val;
}

/** mat_set_init
* Inicialize the matrix of the structure with (i*i)/(mrows-1*mrows-1)
* */
void mat_set_init(Matrix* Mat)
{
    int  i, j, k;
    //float tt;

    for (i = 0; i < Mat->mrows; i++) {
        for (j = 0; j < Mat->mcols; j++) {
            for (k = 0; k < Mat->mdeps; k++) {
                MR(Mat, 0, i, j, k) = (float)(i * i)
                    / (float)((Mat->mrows - 1) * (Mat->mrows - 1));
            }
        }
    }
}

#define SLICE_SIZE 30

int createDevMat(Matrix* dev, int rows, int cols, int deps)
{
    dev->mnums = 1;
    dev->mrows = rows;
    dev->mcols = cols;
    dev->mdeps = deps;
    dev->m = NULL;

    HANDLE_ERROR(cudaMalloc(&(dev->m), dev->mrows * dev->mcols * dev->mdeps * sizeof(float)));

    return(dev->m != NULL) ? 1 : 0;
}

void transferData(Matrix* dest, Matrix* src, int n, int sliceStart, int sliceEnd, cudaMemcpyKind k)
{
    Matrix* dev = (k == cudaMemcpyHostToDevice) ? dest : src;
    Matrix* hst = (k == cudaMemcpyHostToDevice) ? src : dest;

    if (k == cudaMemcpyHostToDevice)
        dev->mrows = (sliceEnd - sliceStart + 1);

    size_t rect = hst->mcols * hst->mdeps;
    size_t transfer_size = dev->mrows * rect * sizeof(float); // dev->mrows = 32
    size_t offset = (n * hst->mrows + sliceStart) * rect;
    float* hst_pos = hst->m + offset;

    switch (k) {
    case cudaMemcpyHostToDevice:
        HANDLE_ERROR(cudaMemcpy(dev->m, hst_pos, transfer_size, cudaMemcpyHostToDevice));
        break;

    case cudaMemcpyDeviceToHost:
        HANDLE_ERROR(cudaMemcpy(hst_pos + rect, dev->m + rect,
            transfer_size - rect * sizeof(float), cudaMemcpyDeviceToHost));
        break;

    }
}

#define THREADPERBLOCK_FORREDUCTION 512

__global__ void add_gosas(float* dev_gosa) {

    __shared__ float cache[THREADPERBLOCK_FORREDUCTION];

    cache[threadIdx.x] = dev_gosa[blockIdx.x * blockDim.x + threadIdx.x]; // each thread safe his value on cache
    __syncthreads();

    //Reduction
    int l = blockDim.x / 2;
    while (l != 0) {
        if (threadIdx.x < l)
            cache[threadIdx.x] += cache[threadIdx.x + l];
        __syncthreads();
        l /= 2;

    }

    if (threadIdx.x == 0) {
        dev_gosa[blockIdx.x] = cache[0]; // return a matrix with gosas that have to be add
    }

}

#define THREADPERBLOCK_COLS 4
#define THREADPERBLOCK_DEPS 128

float jacobi(int nn, Matrix a, Matrix b, Matrix c, Matrix p, Matrix bnd, Matrix wrk1, Matrix wrk2) {

    float* dev_gosa;
    float gosa;
    float* host_gosa;

    int mimax, mjmax, mkmax, box, imax, jmax, kmax;

    mimax = p.mrows; // 32 o 64 o 128 o 256 0 512
    mjmax = p.mcols;
    mkmax = p.mdeps;

    imax = mimax - 1;
    jmax = mjmax - 1;
    kmax = mkmax - 1;
    box = p.mrows * p.mcols * p.mdeps;

    Matrix dev_a0, dev_a1, dev_a2, dev_a3;
    Matrix dev_b0, dev_b1, dev_b2;
    Matrix dev_c0, dev_c1, dev_c2;
    Matrix dev_p, dev_bnd, dev_wrk1, dev_wrk2;

    //Inicializing space on cuda
    createDevMat(&dev_p, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_bnd, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_wrk1, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_wrk2, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_a0, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_a1, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_a2, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_a3, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_b0, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_b1, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_b2, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_c0, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_c1, SLICE_SIZE + 2, mjmax, mkmax);
    createDevMat(&dev_c2, SLICE_SIZE + 2, mjmax, mkmax);

    //Save space on gpu for gosa
    HANDLE_ERROR(cudaMalloc(&dev_gosa, box * sizeof(float)));

    dim3 blocksForReduction = (box / THREADPERBLOCK_FORREDUCTION);
    dim3 threadsPerBlockForReduction = (THREADPERBLOCK_FORREDUCTION);
    host_gosa = (float*)malloc((box / THREADPERBLOCK_FORREDUCTION) * sizeof(float));
    float* swap = (float*)malloc(1 * mimax * mjmax * mkmax * sizeof(float));

    dim3 threadsPerBlock((THREADPERBLOCK_COLS + 2), (THREADPERBLOCK_DEPS + 2));
    dim3 blocks((jmax + THREADPERBLOCK_COLS - 1) / THREADPERBLOCK_COLS,
                   (kmax + THREADPERBLOCK_DEPS - 1)/ THREADPERBLOCK_DEPS);
    printf(" Blocks: {%d, %d, %d} threads. Grid: {%d, %d, %d} blocks.\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, blocks.x, blocks.y, blocks.z);


    for (int n = 0; n < nn; n++) {
        HANDLE_ERROR(cudaMemset(dev_gosa, 0, box * sizeof(float)));
        gosa = 0.f;

        //execute different slices
        // iter 0 -> send 0 to 31
        // iter 1 -> send 30 to 61
        // iter 2 -> send 60 to 91
        // iter 3 -> send 90 to ....
        for (int slice = 0; slice < mimax - 2; slice += SLICE_SIZE) { // desde 0 hasta 29 con paso 30
            int end = min((slice + SLICE_SIZE + 1), imax); // desde 0 a 31 o el final

            //Copy Struct to de GPU like linear array
            transferData(&dev_a0, &a, 0, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_a1, &a, 1, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_a2, &a, 2, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_a3, &a, 3, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_b0, &b, 0, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_b1, &b, 1, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_b2, &b, 2, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_c0, &c, 0, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_c1, &c, 1, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_c2, &c, 2, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_p, &p, 0, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_bnd, &bnd, 0, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_wrk1, &wrk1, 0, slice, end, cudaMemcpyHostToDevice);
            transferData(&dev_wrk2, &wrk2, 0, slice, end, cudaMemcpyHostToDevice);

            Cudajacobi <<<blocks, threadsPerBlock>>> (omega, dev_a0, dev_a1, dev_a2, dev_a3, dev_b0,
                dev_b1, dev_b2, dev_c0, dev_c1, dev_c2, dev_p, dev_bnd, dev_wrk1, dev_wrk2, dev_gosa);

            transferData(&wrk2, &dev_wrk2, 0, slice, end, cudaMemcpyDeviceToHost);

        }

        swap = p.m;
        p.m = wrk2.m;
        wrk2.m = swap;

        //Reduction
        add_gosas <<<blocksForReduction, threadsPerBlockForReduction >>> (dev_gosa);
        HANDLE_ERROR(cudaMemcpy(host_gosa, dev_gosa, (box / THREADPERBLOCK_FORREDUCTION) * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int z = 0; z < (box / THREADPERBLOCK_FORREDUCTION); z++) {
            gosa += host_gosa[z];
        }

    }

    //free
    cudaFree(dev_p.m);
    cudaFree(dev_bnd.m);
    cudaFree(dev_wrk1.m);
    cudaFree(dev_wrk2.m);
    cudaFree(dev_a0.m);
    cudaFree(dev_a1.m);
    cudaFree(dev_a2.m);
    cudaFree(dev_a3.m);
    cudaFree(dev_b0.m);
    cudaFree(dev_b1.m);
    cudaFree(dev_b2.m);
    cudaFree(dev_c0.m);
    cudaFree(dev_c1.m);
    cudaFree(dev_c2.m);
    cudaFree(dev_gosa);
    free(host_gosa);
        
    return gosa;
}



__global__  void Cudajacobi(float omega, Matrix dev_a0, Matrix dev_a1, Matrix dev_a2, Matrix dev_a3,
    Matrix dev_b0, Matrix dev_b1, Matrix dev_b2, Matrix dev_c0, Matrix dev_c1, Matrix dev_c2, Matrix dev_p,
    Matrix dev_bnd, Matrix dev_wrk1, Matrix dev_wrk2, float* dev_gosa) {

    int    i, j, k, imax, jmax, kmax;
    float  s0, ss;
    
    imax = dev_p.mrows - 1;
    jmax = dev_p.mcols - 1;
    kmax = dev_p.mdeps - 1;


    j = threadIdx.x + blockIdx.x * THREADPERBLOCK_COLS;
    k = threadIdx.y + blockIdx.y * THREADPERBLOCK_DEPS;

    //Shared Memory
    __shared__ float p_cache[3][THREADPERBLOCK_COLS + 2][THREADPERBLOCK_DEPS + 2];

    int cache_k = threadIdx.y;
    int cache_j = threadIdx.x;
    int offset = j * dev_p.mdeps + k;

    //Only compute the threads that are not in a front of the block and don´t exceed the matrix
    bool compute = cache_j > 0 && cache_k > 0 && 
                    j < jmax && k < kmax &&
                    cache_j <= THREADPERBLOCK_COLS && cache_k <= THREADPERBLOCK_DEPS;

    for (i = 0; i < imax - 1; i++) { // p.mrows == slice_size + 2
        int row = i * dev_p.mcols * dev_p.mdeps;
        p_cache[0][cache_j][cache_k] = dev_p.m[offset + row];
        p_cache[1][cache_j][cache_k] = dev_p.m[offset + dev_p.mcols * dev_p.mdeps + row];
        p_cache[2][cache_j][cache_k] = dev_p.m[offset + 2*(dev_p.mcols * dev_p.mdeps) + row];

        __syncthreads();

        if (compute) { 

            //it is adding elements of different matrixes to get s0, but the amount of different
            // elements and the distants between them makes the computer to bring a lot of caches
            // blocks to caches to get the s0 result.
            s0 = CMR(dev_a0, i + 1, j, k) * p_cache[2][cache_j][cache_k]
                + CMR(dev_a1, i + 1, j, k) * p_cache[1][cache_j + 1][cache_k]
                + CMR(dev_a2, i + 1, j, k) * p_cache[1][cache_j][cache_k + 1]

                + CMR(dev_b0, i + 1, j, k)
                * (p_cache[2][cache_j + 1][cache_k] - p_cache[2][cache_j - 1][cache_k]
                    - p_cache[0][cache_j + 1][cache_k] + p_cache[0][cache_j - 1][cache_k])

                + CMR(dev_b1, i + 1, j, k)
                * (p_cache[1][cache_j + 1][cache_k + 1] - p_cache[1][cache_j - 1][cache_k + 1]
                    - p_cache[1][cache_j + 1][cache_k - 1] + p_cache[1][cache_j - 1][cache_k - 1])

                + CMR(dev_b2, i + 1, j, k)
                * (p_cache[2][cache_j][cache_k + 1] - p_cache[0][cache_j][cache_k + 1]
                    - p_cache[2][cache_j][cache_k - 1] + p_cache[0][cache_j][cache_k - 1])

                + CMR(dev_c0, i + 1, j, k) * p_cache[0][cache_j][cache_k]
                + CMR(dev_c1, i + 1, j, k) * p_cache[1][cache_j - 1][cache_k]
                + CMR(dev_c2, i + 1, j, k) * p_cache[1][cache_j][cache_k - 1]
                + CMR(dev_wrk1, i + 1, j, k);

            //s0 is used to obtein ss
            ss = (s0 * CMR(dev_a3, i + 1, j, k) - p_cache[1][cache_j][cache_k]) * CMR(dev_bnd, i + 1, j, k);

            //add to gosa ss*ss to get final result, is need to be exclusive access to gosa
            dev_gosa[offset + row] += (ss * ss);
            
            //store on wrk2 = p +omega*ss
            CMR(dev_wrk2, i + 1, j, k) = p_cache[1][cache_j][cache_k] + omega * ss; //Each thread writes a element of the matrix

        }
    }

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