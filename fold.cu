#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <fstream>

using namespace std;

//#define M 1000
//#define N 1000

__global__ void foldkernel(bool* a, bool*c, int m, int n){
	int j=blockIdx.x*blockDim.x+ threadIdx.x;

	if(j<n){
		for(int i=0;i<m;i++)
			c[j] |= a[i*n+j];
	}	
}

int main(int argc, char* argv[]){
	int M,N;
	M = N = atoi(argv[2]);
	int iter = atoi(argv[3]);

	clock_t start, end;
	ifstream in;
	ofstream out;
	in.open(argv[1]);
	long long int size= sizeof(bool) * M * N;
	double gpu_time_used,cpu_time_used;


	bool* mat = (bool*)malloc(size);
	bool* res = (bool*)malloc(sizeof(bool)*N);
	bool *d_mat,*d_res;
	int threadsPerBlock = 512;
	long long int numBlocks = (N+threadsPerBlock-1)/threadsPerBlock;
	for(long long int i=0; i<M; i++){
		for(long long int j=0; j<N; j++)
			in >> mat[i*N+j];
	}
	for(long long int i=0;i<N;i++)
		res[i]=false;

	
	
	start=clock();
	cudaMalloc((void**)&d_mat, size);
	cudaMalloc((void**)&d_res, sizeof(bool)*N);
	cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res, sizeof(bool)*N, cudaMemcpyHostToDevice);
	for(int k=0;k<iter;k++)
		foldkernel<<<numBlocks, threadsPerBlock>>>(d_mat, d_res,M,N);
	cudaMemcpy(res, d_res, sizeof(bool)*N, cudaMemcpyDeviceToHost);	
	end = clock();
	gpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
//	cout << "GPU: " << gpu_time_used << endl;

	start=clock();
	for(int k=0;k<iter;k++){
		for(int j=0; j<N; j++){
			for(int i=0; i<M; i++)
				res[j] |= mat[i*N+j];
		}
	}	
	end = clock();
	
	cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
//	cout << "CPU: " << cpu_time_used << endl;
	cout << "CPU/GPU: " << cpu_time_used/gpu_time_used << endl;	
	return 0;
}
