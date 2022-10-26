/********************************************************************

 This benchmark test program is measuring a cpu performance
 of floating point operation by a Poisson equation solver.

 If you have any question, please ask me via email.
 written by Ryutaro HIMENO, November 26, 2001.
 Version 3.0
 ----------------------------------------------
 Ryutaro Himeno, Dr. of Eng.
 Head of Computer Information Division,
 RIKEN (The Institute of Pysical and Chemical Research)
 Email : himeno@postman.riken.go.jp
 ---------------------------------------------------------------
 You can adjust the size of this benchmark code to fit your target
 computer. In that case, please chose following sets of
 [mimax][mjmax][mkmax]:
 small : 33,33,65
 small : 65,65,129
 midium: 129,129,257
 large : 257,257,513
 ext.large: 513,513,1025
 This program is to measure a computer performance in MFLOPS
 by using a kernel which appears in a linear solver of pressure
 Poisson eq. which appears in an incompressible Navier-Stokes solver.
 A point-Jacobi method is employed in this solver as this method can
 be easyly vectrized and be parallelized.
 ------------------
 Finite-difference method, curvilinear coodinate system
 Vectorizable and parallelizable on each grid point
 No. of grid points : imax x jmax x kmax including boundaries
 ------------------
 A,B,C:coefficient matrix, wrk1: source term of Poisson equation
 wrk2 : working area, OMEGA : relaxation parameter
 BND:control variable for boundaries and objects ( = 0 or 1)
 P: pressure
********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <openacc.h>

#define MR(mt,n,r,c,d)  mt->m[(n) * mt->mrows * mt->mcols * mt->mdeps + (r) * mt->mcols* mt->mdeps + (c) * mt->mdeps + (d)]
/*define preprocessor macro
 * This let us write MR(mt,n,r,c,d) insted of mt->m[(n) * mt->mrows * mt->mcols * mt->mdeps + (r) * mt->mcols* mt->mdeps + (c) * mt->mdeps + (d)]
 *
 * This expresion is use to select the 3D matrix index and fill it. So the mt,n,r,c,and d are variables that are reemplaced in the expresion
 *
 * */
#define CMR(mt,n,r,c,d)  mt[(n) * mimax * mjmax * mkmax + (r) * mjmax * mkmax + (c) * mkmax + (d)]

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
float jacobi(int nn, Matrix* a, Matrix* b, Matrix* c,
    Matrix* p, Matrix* bnd, Matrix* wrk1, Matrix* wrk2);
double fflop(int, int, int);
double mflops(int, double, double);
double second();

//global variables
float   omega = 0.8;
Matrix  a, b, c, p, bnd, wrk1, wrk2, gosa3D;

int main(int argc, char* argv[])
{
    //int    i,j,k,
    int nn;
    int    imax, jmax, kmax, mimax, mjmax, mkmax, msize[3];
    float  gosa, target;
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
    newMat(&p, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&bnd, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&wrk1, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&wrk2, 1, mimax, mjmax, mkmax); // 1 matrixes inicialize with 0.0
    newMat(&a, 4, mimax, mjmax, mkmax); // 4 matrixes inicialize with 0.0
    newMat(&b, 3, mimax, mjmax, mkmax); // 3 matrixes inicialize with 0.0
    newMat(&c, 3, mimax, mjmax, mkmax); // 3 matrixes inicialize with 0.0
    newMat(&gosa3D, 1, imax - 1, jmax - 1, kmax - 1); // 1 matrixes inicialize with 0.0

    mat_set_init(&p); //matrix m of p is empty what MR(Mat,n,r,c,d) do?
    mat_set(&bnd, 0, 1.0); //put MR of bnd matrix 0 to 1.0
    mat_set(&wrk1, 0, 0.0); //put MR of wrk1 matrix 0 to 0.0
    mat_set(&wrk2, 0, 0.0); //put MR of wrk2 matrix 0 to 0.0
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
    mat_set(&gosa3D, 0, 0.0); //put MR of wrk2 matrix 0 to 0.0

    /*
    *    Start measuring
    */
    float *dev_a, *dev_b, *dev_c, *dev_p, *dev_bnd, *dev_wrk1, *dev_wrk2;

    dev_a = a.m;
    dev_b = b.m;
    dev_c = c.m;
    dev_p = p.m;
    dev_bnd = bnd.m;
    dev_wrk1 = wrk1.m;
    dev_wrk2 = wrk2.m;

    #pragma acc data copyin(dev_a) copyin(dev_b) copyin(dev_c) copyin(dev_bnd) copyin(dev_wrk1) copyin(dev_wrk2) 
    #pragma acc data copy(dev_p)
    {
        jacobi(1, &a, &b, &c, &p, &bnd, &wrk1, &wrk2); 

        nn = 3;
        printf(" Start rehearsal measurement process.\n");
        printf(" Measure the performance in %d times.\n\n", nn);

        cpu0 = second(); //get initial time
        gosa = jacobi(nn, &a, &b, &c, &p, &bnd, &wrk1, &wrk2); //use cache bandwith
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
        gosa = jacobi(nn, &a, &b, &c, &p, &bnd, &wrk1, &wrk2);
        cpu1 = second();
        cpu = cpu1 - cpu0;

        printf(" Loop executed for %d times\n", nn);
        printf(" Gosa : %e \n", gosa);
        printf(" MFLOPS measured : %f\tcpu : %f\n", mflops(nn, cpu, flop), cpu);
        printf(" Score based on Pentium III 600MHz using Fortran 77: %f\n",
            mflops(nn, cpu, flop) / 82.84);
    }
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

/** jacobi
 *  Calculate diferents metrics with a huge use of the cache brandwith
 * */
float jacobi(int nn, Matrix* a, Matrix* b, Matrix* c,
    Matrix* p, Matrix* bnd, Matrix* wrk1, Matrix* wrk2)
{
    int    i, j, k, n, imax, jmax, kmax;
    float  gosa, s0, ss;
    float *dev_a, *dev_b, *dev_c, *dev_p, *dev_bnd, *dev_wrk1, *dev_wrk2;

    int mimax = p->mrows;
    int mjmax = p->mcols;
    int mkmax = p->mdeps;

    imax = p->mrows - 1;
    jmax = p->mcols - 1;
    kmax = p->mdeps - 1;

    dev_a = a->m;
    dev_b = b->m;
    dev_c = c->m;
    dev_bnd = bnd->m;
    dev_wrk1 = wrk1->m;
    dev_wrk2 = wrk2->m;
    dev_p = p->m;

    //Copy to GPU already sent
    #pragma acc data present (dev_a, dev_b, dev_c, dev_bnd, dev_wrk1, dev_wrk2, dev_p)
    {

        for (n = 0; n < nn; n++) {
            gosa = 0.0;

            #pragma acc parallel loop reduction(+:gosa)
            for (i = 1; i < imax; i++) {
                #pragma acc loop reduction(+:gosa)
                for (j = 1; j < jmax; j++) {
                    #pragma acc loop reduction(+:gosa)
                    for (k = 1; k < kmax; k++) {
                        //it is adding elements of diferents matrixes to get s0, but the amount of diferents
                        // elements and the distants between them makes the computer to bring a lot of caches
                        // blocks to caches to get the s0 result.
                        s0 = CMR(dev_a, 0, i, j, k) * CMR(dev_p, 0, i + 1, j, k)
                            + CMR(dev_a, 1, i, j, k) * CMR(dev_p, 0, i, j + 1, k)
                            + CMR(dev_a, 2, i, j, k) * CMR(dev_p, 0, i, j, k + 1)
                            + CMR(dev_b, 0, i, j, k)
                            * (CMR(dev_p, 0, i + 1, j + 1, k) - CMR(dev_p, 0, i + 1, j - 1, k)
                                - CMR(dev_p, 0, i - 1, j + 1, k) + CMR(dev_p, 0, i - 1, j - 1, k))
                            + CMR(dev_b, 1, i, j, k)
                            * (CMR(dev_p, 0, i, j + 1, k + 1) - CMR(dev_p, 0, i, j - 1, k + 1)
                                - CMR(dev_p, 0, i, j + 1, k - 1) + CMR(dev_p, 0, i, j - 1, k - 1))
                            + CMR(dev_b, 2, i, j, k)
                            * (CMR(dev_p, 0, i + 1, j, k + 1) - CMR(dev_p, 0, i - 1, j, k + 1)
                                - CMR(dev_p, 0, i + 1, j, k - 1) + CMR(dev_p, 0, i - 1, j, k - 1))
                            + CMR(dev_c, 0, i, j, k) * CMR(dev_p, 0, i - 1, j, k)
                            + CMR(dev_c, 1, i, j, k) * CMR(dev_p, 0, i, j - 1, k)
                            + CMR(dev_c, 2, i, j, k) * CMR(dev_p, 0, i, j, k - 1)
                            + CMR(dev_wrk1, 0, i, j, k);

                        //s0 is used to obtein ss
                        ss = (s0 * CMR(dev_a, 3, i, j, k) - CMR(dev_p, 0, i, j, k)) * CMR(dev_bnd, 0, i, j, k);

                        //add to gosa ss*ss
                        gosa += ss * ss;

                        //store on wrk2 = p +omega*ss
                        CMR(dev_wrk2, 0, i, j, k) = CMR(dev_p, 0, i, j, k) + omega * ss;
                    }
                }
            }

            #pragma acc parallel loop
            for (i = 1; i < imax; i++)
                #pragma acc loop 
                for (j = 1; j < jmax; j++)
                    #pragma acc loop 
                    for (k = 1; k < kmax; k++) //copy wrk2 on p
                        CMR(dev_p, 0, i, j, k) = CMR(dev_wrk2, 0, i, j, k);

        } /* end n loop */
    }

    return(gosa);
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