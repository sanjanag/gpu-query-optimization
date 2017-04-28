#include <iostream>
#include <stdio.h>
#include "bitmat.hpp"
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

using namespace std;

__device__ void Memcpy(void* dest, void* src, size_t n){
    char *csrc = (char *)src;
    char *cdest = (char *)dest;
 
   // Copy contents of src[] to dest[]
   for (int i=0; i<n; i++)
       cdest[i] = csrc[i];
}


__global__ void foldkernel(int* mapping, unsigned char* input, unsigned char* partial, unsigned int num_subs, unsigned int mask_size, unsigned int split){
	unsigned int gap_size = 4;
	unsigned int row_size = 4;

    unsigned int threadid = blockIdx.x * blockDim.x+ threadIdx.x;
    
    unsigned int start_row = threadid * split;
    //int gap_size = sizeof(unsigned int);
    
    for(unsigned int k = start_row; k < (split+start_row); k++){
        unsigned int rowid = k;
        if(rowid >= num_subs)
            break;

        if(mapping[rowid] == -1)
            continue;

        unsigned char* data = input + mapping[rowid];
        
        unsigned int rowsize=0;

        Memcpy(&rowsize, data, row_size);

        data += row_size;

        unsigned int total_cnt = (rowsize-1)/gap_size, cnt = 0, tmpcnt = 0, bitpos = 0, bitcnt = 0;
        
        bool flag = data[0] & 0x01;
        
        unsigned char format = data[0] & 0x02;

        data++;

        while(cnt < total_cnt){
        	
        	Memcpy(&tmpcnt, &data[cnt*gap_size], gap_size);

        	if(format == 0x02){
        		partial[threadid*mask_size + (tmpcnt-1)/8] |= (0x80 >> ((tmpcnt-1)%8));
        	}
        	else{
        		if(flag){
        			for (bitpos = bitcnt; bitpos < bitcnt+tmpcnt; bitpos++) {
						partial[threadid*mask_size + bitpos/8] |= (0x80 >> (bitpos % 8));
					}
        		}
        	}
        	cnt++;
        	flag = !flag;
        	bitcnt += tmpcnt;
        }

        if(format == 0x02){
        	assert((tmpcnt-1)/8 < mask_size);
        }
        else{
        	assert((bitcnt-1)/8 < mask_size);
        }
    }
}

/*void test_fold(unsigned char* cpu_fold, unsigned char* gpu_fold, unsigned int mask_size){
	if(mask_size > 0){
		for(unsigned int i=0; i<mask_size; i++){
			if(cpu_fold[i] != gpu_fold[i]){
				cout << i << " FAIL\n";
				return;
			}
		}
	}
	cout << "PASS\n";
}*/

int main(int argc, char* argv[]){
	//cout << "Hello World!\n";
	cout << "node bitmat_size ratio CPU_time memory_latency kernel_time add_time\n";

	BitMat* bitmat = new BitMat;
	char dumpfile[1024] = "/data/gpuuser1/gpu_query_opt/dump/dbpedia565m_spo_pdump";

	init_bitmat(bitmat, gnum_subs, gnum_preds, gnum_objs, gnum_comm_so, SPO_BITMAT);
	unsigned int node = atoi(argv[1]);
	unsigned int ratio;
	unsigned int triples =  load_from_dump_file(dumpfile, node, bitmat, true, true, NULL, 0, 0, NULL, 0, true);
	
	struct timeval t1, t2;
    double cpu_time, kernel_time, memory_latency, add_time;
    
    gettimeofday(&t1, NULL);
	simple_fold(bitmat, COLUMN, bitmat->objfold, bitmat->object_bytes);	
	gettimeofday(&t2, NULL);
    
    cpu_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    cpu_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
 //   cout << cpu_time << " ms.\n";	
	
    int* mapping = new int[gnum_subs];


	for(unsigned int i=0; i < gnum_subs; i++){
		mapping[i] = -1;
	}

	unsigned long long int gpu_input_size = get_size_of_gpu_input(bitmat);
	float bitmat_size = gpu_input_size;
	bitmat_size = bitmat_size/(1024*1024);


	unsigned char* gpu_input = (unsigned char*)malloc(gpu_input_size * sizeof(unsigned char));
	//cout << "here2\n";
	unsigned int mask_size = bitmat->object_bytes;

	unsigned char *d_partial, *d_input;

	int* d_mapping;
	cudaMalloc((void**)&d_mapping, gnum_subs * sizeof(int));
    cudaMalloc((void**)&d_input, gpu_input_size * sizeof(unsigned char));
    
    gettimeofday(&t1,NULL);
	cudaMemcpy(d_mapping, mapping, gnum_subs * sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, gpu_input, gpu_input_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    gettimeofday(&t2,NULL);
    memory_latency = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    memory_latency += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

	int threadsPerBlock = 512;
    
    convert_bitmat_to_gpu_input(bitmat, gpu_input, mapping, gnum_subs);
    unsigned char* gpu_objfold = (unsigned char*)malloc(mask_size * sizeof(unsigned char));

    for(ratio=40; ratio<=45; ratio+=5){
    	unsigned int split = gnum_subs/ratio;
		unsigned int partial_rows = (gnum_subs%split > 0 ? gnum_subs/split + 1 : gnum_subs/split);
		unsigned int partial_size = partial_rows * mask_size;
		unsigned char* partial = (unsigned char*)malloc(partial_size * sizeof(unsigned char));
		memset(partial, 0 , partial_size);	
		cudaMalloc((void**)&d_partial, partial_size * sizeof(unsigned char));

		gettimeofday(&t1,NULL);
	    cudaMemcpy(d_partial, partial, partial_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	    gettimeofday(&t2,NULL);
	    memory_latency += (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    	memory_latency += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
		
		int numBlocks = (partial_rows%threadsPerBlock > 0 ? partial_rows/threadsPerBlock + 1 : partial_rows/threadsPerBlock);    
		gettimeofday(&t1,NULL);
	    foldkernel<<<numBlocks,threadsPerBlock>>>(d_mapping, d_input, d_partial, gnum_subs, mask_size, split);
	    gettimeofday(&t2,NULL);
	    kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    	kernel_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
	    
	    gettimeofday(&t1,NULL);
	    cudaMemcpy(partial, d_partial, partial_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	    gettimeofday(&t2,NULL);
	    memory_latency += (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    	memory_latency += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

	    memset(gpu_objfold, 0 , mask_size);

	    gettimeofday(&t1,NULL);
	    for(unsigned int i=0; i<partial_rows; i++){
	        for(unsigned int j=0; j<mask_size; j++){
	            gpu_objfold[j] |= partial[i*mask_size + j];
	        }

    	}
    	gettimeofday(&t2,NULL);
		add_time = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    	add_time += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    	cout << node << " " << bitmat_size << " MB " << ratio << " " <<  cpu_time << " ms. "  << memory_latency << " ms. " << kernel_time << " ms. " << add_time << " ms.\n";
    }
	
	
    
    
    //test_fold(bitmat->objfold, gpu_objfold, mask_size);
	return 0;
}