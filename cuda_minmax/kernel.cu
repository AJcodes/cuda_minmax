#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include <math.h>
#include <stdio.h>
#include <random>
#include <iomanip>
#include <iostream>

#define N 16
#define BLOCKSIZE 16

cudaError_t minmaxCuda(double *max, double *min, const double *a, float &time, float &seq_time);

__global__ void minmaxKernel(double *max, double *min, const double *a) {
	__shared__ double maxtile[BLOCKSIZE];
	__shared__ double mintile[BLOCKSIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = a[i];
	mintile[tid] = a[i];
	__syncthreads();
	
	// strided index and non-divergent branch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}

__global__ void seq_minmaxKernel(double *max, double *min, const double *a) {
	__shared__ double maxtile[BLOCKSIZE];
	__shared__ double mintile[BLOCKSIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = a[i];
	mintile[tid] = a[i];
	__syncthreads();
	
	//sequential addressing by reverse loop and thread-id based indexing
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}

__global__ void finalminmaxKernel(double *max, double *min) {
	__shared__ double maxtile[BLOCKSIZE];
	__shared__ double mintile[BLOCKSIZE];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = max[i];
	mintile[tid] = min[i];
	__syncthreads();
	
	// strided index and non-divergent branch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}

__global__ void seq_finalminmaxKernel(double *max, double *min) {
	__shared__ double maxtile[BLOCKSIZE];
	__shared__ double mintile[BLOCKSIZE];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = max[i];
	mintile[tid] = min[i];
	__syncthreads();
	
	//sequential addressing by reverse loop and thread-id based indexing
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}

int main()
{
    const double a[N*N] = {-8.5, -8.4, -6.8, -4.5, -4.2, -3.9, -3.4, -2.3, 1.5, 3.3, 4.3, 4.7, 6.5, 6.7, 8.0, 9.4,
							-7.3, -6.9, -6.0, -4.8, -4.4, -4.3, -3.8, -5.0, 2.5, 2.9, 5.8, 6.3, 6.7, 7.1, 8.0, 9.0,
							-9.0, -8.2, -6.0, -4.8, -1.7, -1.2, -1.0, 2.1, 2.7, 3.1, 4.0, 4.2, 7.3, 7.9, 8.1, 8.8,
							-9.4, -8.5, -7.2, -6.6, -5.1, -4.4, -3.8, -3.1, -1.9, 2.0, 1.7, 2.5, 3.3, 5.1, 5.7, 6.6,
							-9.6, -8.9, -5.9, -2.5, -2.1, -1.8, -8.0, 1.0, 1.7, 2.3, 3.0, 3.8, 5.3, 6.4, 8.4, 9.9,
							-9.7, -8.8, -8.1, -7.5, -4.9, -4.2, -2.2, -6.0, 2.1, 3.3, 3.5, 5.3, 5.8, 5.9, 6.7, 7.2,
							-9.5, -8.8, -8.3, -8.2, -7.1, -6.5, -4.4, -3.6, -1.1, -6.0, 2.5, 3.8, 4.5, 4.7, 7.1, 9.6,
							-9.6, -8.6, -8.4, -6.9, -5.5, -5.4, -4.8, -3.9, -3.6, -7.0, 9.0, 1.1, 3.4, 4.3, 5.8, 10.0,
							-9.7, -9.3, -6.1, -5.9, -4.9, -4.6, -4.2, -4.1, -1.8, 4.0, 1.4, 4.0, 5.0, 5.2, 7.3, 7.7,
							-7.9, -5.5, -5.0, -4.2, -4.1, -3.7, -1.5,  1.9, 4.5, 5.4, 6.1, 6.5, 6.7, 7.7, 8.1, 9.8,
							-8.6, -7.1, -5.3, -5.1, -4.5, -4.1, -2.7, -2.4, -2.1, -1.3, -7.0, 4.4, 6.7, 7.0, 8.2, 9.7,
							-9.2, -8.7, -7.9, -6.9, -6.7, -5.3, -2.6, -2.2, -1.9, -1.1, 4.0, 1.4, 6.9, 7.1, 7.9, 9.5,
							-9.9, -6.0, -4.8, -3.4, 4.0,   7.0,  1.2,  1.6, 4.5, 5.3, 6.5, 7.3, 7.6, 8.0, 9.0, 9.8,
							-9.6, -9.0, -6.7, -6.5, -4.8, -3.0, -2.4,  1.1, 1.2, 1.4, 4.0, 4.5, 4.9, 5.5, 7.0, 7.3,
							-8.5, -7.7, -7.1, -6.0, -5.1, -4.8, -3.7, -2.8, -1.8, -1.4, 2.0, 2.3, 4.8, 5.3, 6.4, 9.2,
							-9.4, -6.7, -5.2, -4.6, -3.2, -2.3, -1.9, -5.0, 2.0, 2.9, 3.2, 4.3, 4.7, 5.1, 6.4, 6.6};
    double *max;
	double *min;
	float time = 0.0f;
	float seq_time = 0.0f;

	max = (double *)malloc((N)*sizeof(double));
	min = (double *)malloc((N)*sizeof(double));

    cudaError_t cudaStatus = minmaxCuda(max, min, a, time, seq_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "minmaxCuda failed!");
        return 1;
    }

	/*for (int i = 0; i < N; i++) {
		std::cout << "Max[" << i << "] = " << max[i] << std::endl;
	}
	std::cout << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "Min[" << i << "] = " << min[i] << std::endl;
	}*/

	std::cout << "Parallel Reduction GPU Implementation" << std::endl;
	std::cout << "Execution Time : " << time / 1000 << " seconds" << std::endl;
	std::cout << "Effective Bandwidth : " << (N*N*sizeof(double)*2) / (time / 1000) << " GB/s" << std::endl;
	std::cout << std::endl;

	std::cout << "Parallel Reduction Sequential Addressing GPU Implementation" << std::endl;
	std::cout << "Execution Time : " << seq_time / 1000 << " seconds" << std::endl;
	std::cout << "Effective Bandwidth : " << (N*N*sizeof(double)*2) / (seq_time / 1000) << " GB/s" << std::endl;
	std::cout << std::endl;

	std::cout << "Max value: " << max[0] << std::endl;
	std::cout << "Min value: " << min[0] << std::endl;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t minmaxCuda(double *max, double *min, const double *a, float &time, float &seq_time)
{
    double *dev_a = 0;
    double *dev_max = 0;
	double *dev_min = 0;
	float milliseconds = 0;
	float milliseconds1 = 0;
	dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(N);
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_max, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_min, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, N * N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaEventRecord(start);
    minmaxKernel<<<dimGrid, dimBlock>>>(dev_max, dev_min, dev_a);
	cudaThreadSynchronize();
	finalminmaxKernel<<<1, dimBlock>>>(dev_max, dev_min);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaThreadSynchronize();

	cudaEventRecord(start1);
    seq_minmaxKernel<<<dimGrid, dimBlock>>>(dev_max, dev_min, dev_a);
	cudaThreadSynchronize();
	seq_finalminmaxKernel<<<1, dimBlock>>>(dev_max, dev_min);
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	cudaThreadSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "minmaxKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching minmaxKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(max, dev_max, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(min, dev_min, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventElapsedTime(&milliseconds1, start1, stop1);
	time = milliseconds;
	seq_time = milliseconds1;

Error:
    cudaFree(dev_max);
	cudaFree(dev_min);
    cudaFree(dev_a);
    
    return cudaStatus;
}
