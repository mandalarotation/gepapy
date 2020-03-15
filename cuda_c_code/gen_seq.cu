#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void generator_sequence(double *X, int r, int c){
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < r && col < c)
	{
		X[row*c + col] = col;

	}
}

extern "C" {

	void cuda_generator_sequence(double *R_h, int r_h, int c_h)
	{
		
	// Allocate memory space on the device
	double *R_d;
	cudaMalloc((void **) &R_d, sizeof(double)*r_h*c_h);
	
	unsigned int BLOCK_SIZE = 16;
	unsigned int grid_rows = (r_h + BLOCK_SIZE - 1)/BLOCK_SIZE;
	unsigned int grid_cols = (c_h + BLOCK_SIZE - 1)/BLOCK_SIZE;
	dim3 dimGrid(grid_cols,grid_rows);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	
	generator_sequence<<<dimGrid,dimBlock>>>(R_d,r_h,c_h);
	
	// Copy array back to host
	cudaMemcpy( R_h, R_d, sizeof(double)*r_h*c_h, cudaMemcpyDeviceToHost );
	
	// Release device memory
	cudaFree(R_d);	
	} 
}



