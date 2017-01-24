#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <fstream>

using namespace std;

//#define M 1000
//#define N 1000

__global__ void unfoldkernel(bool* a, bool*mask, bool* c, int m, int n){
	int i = blockIdx.x*blockDim.x+ threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i<m && j<n)
		c[i*n+j] = a[i*n+j]&mask[j];
/*	if(i<m){
		for(int j=0;j<n;j++)
			c[i*n+j] = a[i*n+j]&mask[j];
	}*/
}

int main(int argc, char* argv[]){

	clock_t start, end;
	ifstream in;
	in.open(argv[1]);
	int M,N;
	M = N = atoi(argv[2]);
	int iter = atoi(argv[3]);
	long long int size= sizeof(bool) * M * N;
	double gpu_time_used, cpu_time_used;

	bool* mat = (bool*)malloc(size);
	bool* mask = (bool*)malloc(sizeof(bool)*N);
	bool* res = (bool*)malloc(size);

	bool *d_mat, *d_mask, *d_res;
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks((M+threadsPerBlock.x-1)/threadsPerBlock.x, (N+threadsPerBlock.y-1)/threadsPerBlock.y);
//	int threadsPerBlock = 100;
//	int numBlocks = (M+threadsPerBlock-1)/threadsPerBlock;
	for(long long int i=0; i<M; i++){
		for(long long int j=0; j<N; j++)
			in >> mat[i*N+j];
	}
	for(long long int i=0;i<N;i++)
		in >> mask[i];

	
	
	start = clock();
	cudaMalloc((void**)&d_mat, size);
	cudaMalloc((void**)&d_mask, sizeof(bool)*N);
	cudaMalloc((void**)&d_res, size);

	cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, sizeof(bool)*N, cudaMemcpyHostToDevice);
	for(int k=0; k<iter; k++)
		unfoldkernel<<<numBlocks, threadsPerBlock>>>(d_mat, d_mask,d_res,M,N);
	cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);	

	end = clock();
	gpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
//	cout << "GPU: " << time_used << endl;

	start=clock();
	for(int k=0; k<iter; k++){
		for(int j=0; j<N; j++){
			for(int i=0; i<M; i++)
				res[j] = mat[i*N+j]&mask[j];
		}
	}	
	end = clock();
	
	cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
//	cout << "CPU: " << time_used << endl;
	cout << "CPU/GPU: " << cpu_time_used/gpu_time_used << endl;
/*	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++)
			cout << mat[i*N+j] << '\t';
		cout << endl;
	}
	cout << endl;
	for(int i=0;i<N;i++)
		cout << mask[i] << '\t';
	cout << endl << endl;
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++)
			cout << res[i*N+j] << '\t';
		cout << endl;
	}
*/
	return 0;
}
