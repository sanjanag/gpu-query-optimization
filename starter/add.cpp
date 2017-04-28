#include <iostream>
#include <cuda.h>
#include <cstdlib>

using namespace std;

#define N 2048*2048
#define THREADS_PER_BLOCK 512

__global__ void add(int* a, int* b, int* c){

	int index= threadIdx.x + blockDim.x*blockIdx.x;
	c[index] = a[index] + b[index];

}


int main(){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int)*N;
	
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	for(int i=0;i<N;i++)
		a[i]=rand();

	for(int i=0;i<N;i++)
		b[i]=rand();

	cudaMalloc((void**)&d_a, size); 
	cudaMalloc((void**)&d_b, size); 	
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a,d_b,d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	free(a);
	free(b);
	free(c);
	cudaFree(d_a); 
	cudaFree(d_b); 
	cudaFree(d_c); 
	
	return 0;	

}
