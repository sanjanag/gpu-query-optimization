#include <iostream>
#include <string>
#include "helper.h"
#include <stdio.h>

using namespace std;

__device__ void Memcpy(void* dest, void* src, size_t n){
    char *csrc = (char *)src;
    char *cdest = (char *)dest;
 
   // Copy contents of src[] to dest[]
   for (int i=0; i<n; i++)
       cdest[i] = csrc[i];
}


__global__ void foldkernel(unsigned long long int* mapping, unsigned char* input, unsigned char* res, int n,int size_mask, int split){
	unsigned int GAP_SIZE_BYTES = sizeof(unsigned int);
	unsigned int ROW_SIZE_BYTES = sizeof(unsigned int);
    int threadid = blockIdx.x*blockDim.x+ threadIdx.x;
    int i = threadid*split;
    //int gap_size = sizeof(unsigned int);
    
    for(int k = 0; k < split; k++){
        int rowid = i+k;
        if(rowid >= n)
            break;
        if(mapping[rowid]==-1)
            continue;
        unsigned char* data = input+mapping[rowid];
        unsigned rowsize=0;
        Memcpy(&rowsize, data, ROW_SIZE_BYTES);
        data += ROW_SIZE_BYTES;
        unsigned int total_cnt = (rowsize-1)/GAP_SIZE_BYTES;
        unsigned int cnt = 0;
        unsigned int bitpos = 0, bitcnt = 0;
        bool flag = data[0] & 0x01;
        unsigned char format = data[0] & 0x02;

        data++;

        while(cnt < total_cnt){
        	unsigned int tmpcnt = 0;
        	Memcpy(&tmpcnt, data[cnt*GAP_SIZE_BYTES], GAP_SIZE_BYTES);
        	if(format == 0x02){
        		res[threadid*size_mask + (tmpcnt-1)/8] |= (0x80 >> ((tmpcnt-1)%8));
        	}
        	else{
        		if(flag){
        			for (bitpos = bitcnt; bitpos < bitcnt+tmpcnt; bitpos++) {
						res[threadid*size_mask + bitpos/8] |= (0x80 >> (bitpos % 8));
					}
        		}
        	}
        	cnt++;
        	flag = !flag;
        	bitcnt += tmpcnt;
        }

        if(format == 0x02){
        	assert((tmpcnt-1)/8 < size_mask);
        }
        else{
        	assert((bitcnt-1)/8 < size_mask);
        }
    }
}


int main(){
	BitMat* bitmat = new BitMat;
	unsigned int node = 1;
	char dumpfile[1024] = "/work/scratch/datasets/dbpedia/all_data_bitmats/dbpedia565m_pos_odump";
	
	gnum_subs=20;
	gnum_preds=57453;
	gnum_objs=153561757;
	gnum_comm_so=27116793;

	init_bitmat(bitmat, gnum_subs, gnum_preds, gnum_objs, gnum_comm_so, POS_BITMAT);

	unsigned int triples =  load_from_dump_file(dumpfile, node, bitmat, true, true, NULL, 0, 0, NULL, 0, true);
	//print_bitmat(bitmat->bm);
	//cout << count_size_of_bitmat(bitmat) << endl;	
	//simple_fold(bitmat, COLUMN, bitmat->objfold, bitmat->object_bytes);
	
	unsigned long long int mapping[gnum_subs];
    
    for(int i=0; i<gnum_subs; i++)
        mapping[i] = i;

    unsigned long long int size_gpu_input = get_sizeof_1dimarr(bitmat->bm);
    unsigned char* gpu_input = (unsigned char*)malloc(size_gpu_input*sizeof(unsigned char));
    convert_bitmat_to_1dimarr(bitmat->bm, mapping, gnum_subs, gpu_input);
    int n = gnum_subs;
    int split = 8;
    int rows_res = (n+split-1)/split;
    int size_res = rows_res*size_mask;
    int* d_mapping;
    unsigned char* d_input;
    unsigned char* d_res;
    unsigned char* res = (unsigned char*)malloc(size_res*sizeof(unsigned char));
    for(int i=0; i<rows_res; i++){
        for(int j=0;j<size_mask;j++)
            res[i*size_mask+j] = 0x00;
    }
    
    //cout << "done\n";


    cudaMalloc((void**)&d_mapping, sizeof(int)*n);
    cudaMalloc((void**)&d_input, size_gpu_input*sizeof(unsigned char));
    cudaMalloc((void**)&d_res, size_res*sizeof(unsigned char));


    cudaMemcpy(d_mapping, mapping, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, gpu_input, sizeof(unsigned char)* size_gpu_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, sizeof(unsigned char)*size_res, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int numBlocks = (rows_res+threadsPerBlock-1)/threadsPerBlock;
    //cout << rows_res << '\t' << size_mask << '\t' << threadsPerBlock << '\t' << numBlocks << endl;
    foldkernel<<<numBlocks,threadsPerBlock>>>(d_mapping, d_input, d_res, n, size_mask, split);

    cudaMemcpy(res, d_res, size_res*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    unsigned char* gpu_mask = (unsigned char*)malloc(size_mask*sizeof(unsigned char));
    for(int i=0;i<size_mask;i++){
        gpu_mask[i] = 0x00;
    }

    for(int i=0;i<rows_res; i++){
        for(int j=0;j<size_mask;j++){
            gpu_mask[j] |= res[i*size_mask+j];
        }
    }
	
	return 0;
}