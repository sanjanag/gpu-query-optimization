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
	int threadid = blockIdx.x*blockDim.x+ threadIdx.x;
	int i = (blockIdx.x*blockDim.x+ threadIdx.x)*4;
	
	for(int k=0; k<4; k++){
		if(i+k >= m)
			break;
		for(int j=0; j<n;j++ ){
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
	bool* res = (bool*)malloc(sizeof(bool)*((M+4-1)/4)*N);
	bool* mask = (bool*)malloc(sizeof(bool)*N);

	bool *d_mat,*d_res;
	int threadsPerBlock = 10;
	long long int numBlocks = (M+threadsPerBlock-1)/threadsPerBlock;

	for(long long int i=0; i<M; i++){
		for(long long int j=0; j<N; j++)
			in >> mat[i*N+j];
	}

	for(long long int i=0;i<((M+4-1)/4);i++)
		for(int j=0;j<N;j++)
			res[i*N+j]=false; 
	for(int i=0;i<N;i++)
		mask[i]=false;	

	//print(mat, M, N);
	start=clock();

	cudaMalloc((void**)&d_mat, size);
	cudaMalloc((void**)&d_res, sizeof(bool)*((M+4-1)/4)*N);
	cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res, sizeof(bool)*((M+4-1)/4)*N, cudaMemcpyHostToDevice);

	for(int k=0;k<iter;k++)
		foldkernel<<<numBlocks, threadsPerBlock>>>(d_mat, d_res,M,N);

	cudaMemcpy(res, d_res, sizeof(bool)*((M+4-1)/4)*N, cudaMemcpyDeviceToHost);	
	for(int k=0;k<iter;k++){
		for(int i=0;i<((M+4-1)/4);i++){
			for(int j=0;j<N;j++)
				mask[j] |= res[i*N+j];
		}
	}	
	end = clock();
	//print(res,((M+4-1)/4),N);
	//print(mask,1,N);
	gpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
//	cout << "GPU: " << gpu_time_used << endl;
	start=clock();
	for(int k=0;k<iter;k++){
		for(int i=0;i<M;i++){
			for(int j=0;j<N;j++)
				mask[j] |= mat[i*N+j];
		}
	}	
	end = clock();
	//print(mask,1,N);	
	cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
//	cout << "CPU: " << cpu_time_used << endl;
	cout << "CPU/GPU: " << cpu_time_used/gpu_time_used << endl;	
	return 0;
}
