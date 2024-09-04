// Example matrix addition with GPU -  RIN - 9/4/2024.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>

#define DIM 8
#define N 64
#define BIGN 1024

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void
AddGPUKernel(const float *C, float *SUM)
{
    int k = blockIdx.x*blockDim.x+threadIdx.x;

    if (k < N) SUM[k] += C[k];
}

float
vectorAddGPU(float *hA)
{
   float *A, *SUM;

   float *hSum = (float*)malloc(sizeof(float));
   hSum[0] = 0;

   float *retval = (float*)malloc(sizeof(float));;
   retval[0] = 0;

   int thr_per_blk = 256;
   int blk_in_grid = ceil( float(BIGN) / thr_per_blk );

   // Allocate device and host memory
   cudaMalloc(&A, N * sizeof(float));
   cudaMalloc(&SUM, N * sizeof(float));
   cudaCheckErrors("Allocate device memory");

   // Copy the host vectors to device vectors
   cudaMemcpy(A, &hA, N * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(SUM, &hSum, N * sizeof(float), cudaMemcpyHostToDevice);
   cudaCheckErrors("Mem copy to device");

   // Start timer
   auto start_gpu = std::chrono::high_resolution_clock::now();

   // Perform the kernel on the device
   AddGPUKernel <<<  blk_in_grid, thr_per_blk >>> (A, SUM);
   cudaCheckErrors("Add kernel call");

   // Stop timer
   auto stop_gpu = std::chrono::high_resolution_clock::now();
   auto duration_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_gpu - start_gpu);
   std::cout << "GPU Time   : " << duration_gpu.count() << " ns \n";

   // Synchonize device
   cudaDeviceSynchronize();
   cudaCheckErrors("Failure to synchronize device");

   // Now get the results back to the host
   cudaMemcpy(&hSum, SUM, N*sizeof(float), cudaMemcpyDeviceToHost);
   cudaCheckErrors("mem copy to host");

   // Cleanup and finish up
   cudaFree(SUM);
   cudaFree(A);

   free(retval);
   free(hSum);

   return 0;
}

float
vectorAddCPU(float *hA)
{
   float sum = 0;

   // Start timer
   auto start_cpu = std::chrono::high_resolution_clock::now();

   // Naive implementation
   for (int i=0;i<N;i++) {sum += hA[i];}

   // Stop timer
   auto stop_cpu = std::chrono::high_resolution_clock::now();
   auto duration_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_cpu - start_cpu);
   std::cout << "CPU Time   : " << duration_cpu.count() << " ns \n";

   return sum;
}

int main()
{
        float d3[DIM][DIM]={{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0},{9,10,11,12,13,14,15,16},{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0},{9,10,11,12,13,14,15,16},
                        {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0},{9,10,11,12,13,14,15,16},{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0},{9,10,11,12,13,14,15,16}
};

        printf("d3 matrix :\n\n");

        for (int i=0;i<DIM;i++)
        {
                for (int j=0;j<DIM;j++)
                     printf("%4.2f ",d3[i][j]);
                printf("\n\n");
        }

        printf("Result GPU : %4.2f \n\n",vectorAddGPU((float *)d3));

        printf("Result CPU : %4.2f \n\n",vectorAddCPU((float *)d3));

        printf("\nExecution OVER...\n");

        return 0;
}

