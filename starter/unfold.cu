#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <fstream>

using namespace std;


__global__ void unfoldkernel(bool* a, bool*mask, bool* c, int n){
	int i = blockIdx.x*blockDim.x+ threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i<n && j<n)
		c[i*n+j] = a[i*n+j]&mask[j];
}

void print(bool* a, int m, int n){
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++)
			cout << a[i*n+j] << '\t';
		cout << endl;
	}
	cout << endl;
}

int main(int argc, char* argv[]){
	//Initialisation variables
	ifstream in;
	in.open(argv[1]);
	int n= atoi(argv[2]);
	int iterations = atoi(argv[3]);
	
	//clock variables
	clock_t start, end;
	double gpu_time_used, cpu_time_used;
	
	//Threads and block configuration
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks((n+threadsPerBlock.x-1)/threadsPerBlock.x, (n+threadsPerBlock.y-1)/threadsPerBlock.y);
	
	//sizes of matrices
	int size_mat = sizeof(bool) * n * n;
	int size_res = size_mat;
	int size_mask = sizeof(bool)*n;

	//memory allocation in host machine
	bool* mat = (bool*)malloc(size_mat);
	bool* res = (bool*)malloc(size_res);
	bool* mask = (bool*)malloc(size_mask);

	//Initializing matrices
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++)
			in >> mat[i*n+j];
	}
	for(int i = 0; i < n; i++)
		in >> mask[i];

	bool *d_mat, *d_mask, *d_res;

	start = clock();

	//Memory allocation in GPU
	cudaMalloc((void**)&d_mat, size_mat);
	cudaMalloc((void**)&d_mask, size_mask);
	cudaMalloc((void**)&d_res, size_res);

	//copy data from host to GPU
	cudaMemcpy(d_mat, mat, size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);

	for(int k=0; k<iterations; k++)
		unfoldkernel<<<numBlocks, threadsPerBlock>>>(d_mat, d_mask,d_res, n);

	//copying result
	cudaMemcpy(res, d_res, size_res, cudaMemcpyDeviceToHost);

	end = clock();
	
	//calculating time taken by GPU
	gpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;

	//CPU computation
	start=clock();
	for(int k = 0; k < iterations; k++){
		for(int j = 0; j < n; j++){
			for(int i = 0; i < n; i++)
				res[j] = mat[i*n+j]&mask[j];
		}
	}	
	end = clock();
	
	//calculating time taken by CPU
	cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
	cout << "CPU/GPU: " << cpu_time_used/gpu_time_used << endl;
	return 0;
}
