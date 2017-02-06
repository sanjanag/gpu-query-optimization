#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <fstream>

using namespace std;


__global__ void foldkernel(bool* a, bool*c, int n, int split){
	int threadid = blockIdx.x*blockDim.x+ threadIdx.x;
	int i = (blockIdx.x*blockDim.x+ threadIdx.x)*split;
	
	for(int k = 0; k < split; k++){
		if(i+k >= n)
			break;
		for(int j = 0; j < n; j++){
			c[threadid*n+j] |= a[(i+k)*n+j];
		}
	}
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
	in.open(argv[1]);  //data_file
	int n = atoi(argv[2]); //dimension of bitmat
	int split = atoi(argv[3]);
	int iterations = atoi(argv[4]); //Iterations to compare the results

	//clock variables
	clock_t start, end;
	double gpu_time_used,cpu_time_used;
	
	//Threads and block configuration
	int rows_res = (n+split-1)/split;
	int threadsPerBlock = 512;
	long long int numBlocks = (rows_res+threadsPerBlock-1)/threadsPerBlock;

	//sizes of matrices
	int size_mat = sizeof(bool) * n * n;
	int size_res = sizeof(bool) * (rows_res) * n;
	int size_mask = sizeof(bool) * n;

	//memory allocation in host machine
	bool* mat = (bool*)malloc(size_mat);
	bool* res = (bool*)malloc(size_res);
	bool* mask = (bool*)malloc(size_mask);
	
	//Initializing matrices
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++)
			in >> mat[i*n+j];
	}

	for(int i = 0; i < rows_res; i++)
		for(int j = 0; j < n; j++)
			res[i*n+j]=false; 
	for(int i = 0; i < n; i++)
		mask[i]=false;	
	

	bool *d_mat,*d_res;

	start=clock();

	//Memory allocation in GPU
	cudaMalloc((void**)&d_mat, size_mat);
	cudaMalloc((void**)&d_res, size_res);

	//copy data from host to GPU
	cudaMemcpy(d_mat, mat, size_mat, cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res, size_res, cudaMemcpyHostToDevice);

	for(int k = 0; k < iterations; k++)
		foldkernel<<<numBlocks, threadsPerBlock>>>(d_mat, d_res, n, split);
	
	//copying result
	cudaMemcpy(res, d_res, size_res, cudaMemcpyDeviceToHost);	

	//computing final result
	for(int k = 0; k < iterations; k++){
		for(int i = 0; i < rows_res; i++){
			for(int j = 0; j < n; j++)
				mask[j] |= res[i*n+j];
		}
	}	
	end = clock();

	//calculating time taken by GPU
	gpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
	

	//CPU computation
	start=clock();
	for(int k = 0; k < iterations; k++){
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++)
				mask[j] |= mat[i*n+j];
		}
	}	
	end = clock();
	//calculating time taken by CPU
	cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
	cout << "CPU/GPU: " << cpu_time_used/gpu_time_used << endl;	
	return 0;
}
