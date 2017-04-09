#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cuda.h>

using namespace std;

__device__ void Memcpy(void* dest, void* src, size_t n){
	char* csrc = (char*)src;
	char* cdest = (char*)dest;

	for(int i=0;i <n; i++){
		cdest[i] = csrc[i];
	}
}

__global__ void copykernel(unsigned char* input, unsigned char* output, int n){
	int row = threadIdx.x + blockIdx.x*blockDim.x;
	if(row<n){
		unsigned char* dest = output+row*sizeof(unsigned int);
		unsigned char* src = input+row*sizeof(unsigned int);
		Memcpy(dest, src, sizeof(unsigned int));
	}
}

int main(int argc, char* argv[]){

	int n = atoi(argv[1]);

	unsigned int a[n];
	for(int i=0; i<n; i++)
		a[i] = i+1;

	unsigned char* data = (unsigned char*)malloc(n*sizeof(unsigned int)*sizeof(unsigned char));
	unsigned char* output = (unsigned char*)malloc(n*sizeof(unsigned int)*sizeof(unsigned char));
	unsigned char* curr = data;
	
	for(int i=0;i<n;i++){
		memcpy(curr, &a[i], sizeof(unsigned int));
		curr += sizeof(unsigned int);
	}


	unsigned char* d_input;
	unsigned char* d_output;

	cudaMalloc((void**)&d_input, n*sizeof(unsigned int)*sizeof(unsigned char));
	cudaMalloc((void**)&d_output, n*sizeof(unsigned int)*sizeof(unsigned char));

	cudaMemcpy(d_input, data, n*sizeof(unsigned int)*sizeof(unsigned char), cudaMemcpyHostToDevice);


	int threadsPerBlock = 512;
    int numBlocks = (n+threadsPerBlock-1)/threadsPerBlock;
	

    copykernel<<<threadsPerBlock, numBlocks>>>(d_input,d_output, n);

    cudaMemcpy(output, d_output, n*sizeof(unsigned int)*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
    	unsigned int tmp;
    	memcpy(&tmp, output, sizeof(unsigned int));
    	cout << tmp << endl;
    	output+= sizeof(unsigned int);
    }

	return 0;
}