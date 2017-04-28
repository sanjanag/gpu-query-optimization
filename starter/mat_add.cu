#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <time.h>

using namespace std;

#define M 40000
#define N 40000

__global__ void addkernel(bool* a, bool* b, bool*c, int m, int n){
	int i=blockIdx.x*blockDim.x+ threadIdx.x;
	int j=blockIdx.y*blockDim.y+ threadIdx.y;

	if(i<m && j<n)	
		c[i*n+j] = a[i*n+j] ^ b[i*n+j];
}

int main(){

	clock_t start, end;

	int size= sizeof(bool) * M * N;
	
	bool* mat1 = (bool*)malloc(size);
	bool* mat2 = (bool*)malloc(size);
	bool* res = (bool*)malloc(size);


	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++)
			mat1[i*N+j]=rand()%2;
	}
	
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++)
			mat2[i*N+j]=rand()%2;
	}

	bool *d_mat1, *d_mat2, *d_res;

	
	dim3 threadsPerBlock(16, 32);
	dim3 numBlocks((M+threadsPerBlock.x-1)/threadsPerBlock.x,(N+threadsPerBlock.y-1)/threadsPerBlock.y);
	
	cudaMalloc((void**)&d_mat1, size);
	cudaMalloc((void**)&d_mat2, size);
	cudaMalloc((void**)&d_res, size);
	
	start=clock();
	cudaMemcpy(d_mat1, mat1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat2, mat2, size, cudaMemcpyHostToDevice);
	
	for(int j=0; j<=100; j++)
{
		addkernel<<<numBlocks, threadsPerBlock>>>(d_mat1,d_mat2,d_res,M,N);
}
	cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);	
	end = clock();
	double cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
	cout << "GPU: " << cpu_time_used << endl;

	

/*	start=clock();
	for (int k = 0; k< 10; k++){
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++)
			res[i*N+j] = mat1[i*N+j]^ mat2[i*N+j];
	}
	}
	end = clock();
	cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
	cout << "CPU: " << cpu_time_used << endl;*/
	return 0;
}
