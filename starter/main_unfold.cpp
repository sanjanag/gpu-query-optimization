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

__device__ unsigned long count_bits_in_row(unsigned char *in, unsigned int size)
{
    if (size == 0)
        return 0;

    unsigned int count;
    unsigned int i;
    count = 0;
    for (i = 0; i < size; i++) {
        if (in[i] == 0xff) {
            count += 8;
        } else if (in[i] > 0x00) {
            for (unsigned short j = 0; j < 8; j++) {
                if (in[i] & (0x80 >> j)) {
                    count++;
                }
            }
        }
    }

    return count;
}

__global__ void unfoldkernel(int* mapping, unsigned char* input, unsigned char* mask, unsigned char* output, int n, int size_andres, int size_mask, int num_objs){
    
    int row = threadIdx.x + blockIdx.x*blockDim.x;
    if(row < n){
        if(mapping[row] == -1)
            return;

        unsigned int maskarr_bits = count_bits_in_row(mask, size_mask);

        if (maskarr_bits == num_objs) {
            //pass
           // printf("All objects present.. didn't unfold anything\n");
            return;
        }
        if(size_mask==0){
            return;
        }

        unsigned char* andres = output + size_andres*row; //grave mistake here
        int gap_size = sizeof(unsigned int);
        unsigned int andres_size = 0;
        unsigned char* data  = input+mapping[row];
        unsigned int rowsize = 0;
        Memcpy(&rowsize, data, gap_size);

        data += gap_size;
        unsigned int cnt = 0, total_cnt = (rowsize-1)/gap_size, tmpcnt = 0, triplecnt = 0,
                    prev1_bit = 0, prev0_bit = 0, tmpval = 0;
        bool flag = data[0] & 0x01, begin = true, p1bit_set = false, p0bit_set = false;
        unsigned char format = data[0] & 0x02;

        if(format == 0x02){
            begin = false;
            andres[gap_size] = 0x02;
            andres_size = gap_size+1;
        }

        while(cnt < total_cnt){

            Memcpy(&tmpcnt, &data[cnt*gap_size+1], gap_size);

            if(format==0x02){
                if((tmpcnt-1)/8 >= size_mask){
                    //pass
                }
                else if(mask[(tmpcnt-1)/8] & (0x80 >> ((tmpcnt-1)%8))){
                    Memcpy(&andres[andres_size], &tmpcnt, gap_size);
                    andres_size += gap_size;
                }
            }
            else if (format == 0x00){
                if(flag){
                    if (triplecnt/8 >= size_mask) {
                        prev0_bit = triplecnt + tmpcnt - 1;
                
                    else{
                        for(unsigned int i = triplecnt; i < triplecnt + tmpcnt; i++){
                            if(i/8 >= size_mask){
                                assert(p0bit_set || p1bit_set);
                                if (!p0bit_set && p1bit_set) {
                                    tmpval = prev1_bit + 1;
                                    Memcpy(&andres[andres_size], &tmpval, gap_size);
                                    andres_size += gap_size;
                                    p0bit_set = true;
                                } 
                                else if (prev1_bit > prev0_bit) {
                                    tmpval = prev1_bit - prev0_bit;
                                    Memcpy(&andres[andres_size], &tmpval, gap_size);
                                    andres_size += gap_size;
                                }
                                prev0_bit = i;
                            }
                            else if((mask[i/8] & (0x80 >> (i%8))) == 0x00){
                                if(begin){
                                    begin = false;
                                    andres[gap_size] = 0x00;
                                    andres_size = gap_size + 1;
                                    p0bit_set = true;
                                }
                                else{
                                    if(!p0bit_set){
                                        Memcpy(&andres[andres_size], &i, gap_size);
                                        andres_size += gap_size;
                                        p0bit_set = true;
                                    }
                                    else if(prev0_bit != (i-1)){
                                        tmpval = i - prev0_bit - 1;
                                        Memcpy(&andres[andres_size], &tmpval, gap_size);
                                        andres_size += gap_size;
                                    }
                                }
                                prev0_bit = i;
                            }
                            else if(mask[i/8] & (0x80 >> (i%8))){
                                if(begin){
                                    begin = false;
                                    andres[gap_size] = 0x80;
                                    andres_size = gap_size + 1;
                                    p1bit_set = true;
                                }
                                else{
                                    if(!p1bit_set){
                                        Memcpy(&andres[andres_size], &i, gap_size);
                                        andres_size += gap_size;
                                        p1bit_set = true;
                                    }
                                    else if(prev1_bit != (i-1)){
                                        tmpval = i - prev1_bit - 1;
                                        Memcpy(&andres[andres_size], &tmpval, gap_size);
                                        andres_size += gap_size;
                                    }
                                }
                                prev1_bit = i;
                            }
                        }
                    }
                }
                else{
                    if(begin){
                        begin = false;
                        andres[gap_size] = 0x00;
                        andres_size = gap_size+1;
                        p0bit_set = true;
                    }
                    else{
                        if(!p0bit_set){
                            tmpval = prev1_bit + 1;
                            Memcpy(&andres[andres_size], &tmpval, gap_size);
                            andres_size += gap_size;
                            p0bit_set = true;
                        }
                        else if(prev0_bit != (triplecnt-1)){
                            tmpval = triplecnt -prev0_bit - 1;
                            Memcpy(&andres[andres_size], &tmpval, gap_size);
                            andres_size += gap_size;
                        }
                    }
                    prev0_bit = triplecnt + tmpcnt - 1;
                }

            }
            else{
                assert(0);
            }
            
            flag = !flag;
            cnt++;
            triplecnt += tmpcnt;
        }

        if(format == 0x00){
            if(prev1_bit > prev0_bit){
                if(!p0bit_set){
                    tmpval = prev1_bit + 1;
                    Memcpy(&andres[andres_size], &tmpval, gap_size);
                }
                else{
                    tmpval = prev1_bit - prev0_bit;
                    Memcpy(&andres[andres_size], &tmpval, gap_size);
                }
                andres_size +=  gap_size;
            }
            else{
                if(!p1bit_set){
                    tmpval = prev0_bit + 1;
                    Memcpy(&andres[andres_size], &tmpval, gap_size);                
                }
                else{
                    tmpval = prev0_bit - prev1_bit;
                    Memcpy(&andres[andres_size], &tmpval, gap_size);
                }
                andres_size += gap_size;
            }    
        }
        

        tmpval = andres_size - gap_size;
        Memcpy(andres, &tmpval, gap_size);
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
    int n = gnum_subs;

    int mapping[n];
    for(int i=0; i<n; i++)
        mapping[i] = -1;

    int size_gpu_input = get_sizeof_1dimarr(bm);
    unsigned char* gpu_input = (unsigned char*)malloc(size_gpu_input*sizeof(unsigned char));
    convert_bitmat_to_1dimarr(bm, mapping, n, gpu_input);
    //int gap_size = sizeof(unsigned int);
    int size_gpu_output =  (GAP_SIZE_BYTES * 2 * size_mask + ROW_SIZE_BYTES + 1 + 1024)*n;
    unsigned char* gpu_output = (unsigned char*)malloc(size_gpu_output*sizeof(unsigned char));
    for(int i=0;i<size_gpu_output ;i++)
        gpu_output[i] = 0x00;
 
    int* d_mapping;
    unsigned char* d_input;
    unsigned char* d_mask;
    unsigned char* d_output;
 
    cudaMalloc((void**)&d_mapping, sizeof(int)*n);
    cudaMalloc((void**)&d_input, size_gpu_input*sizeof(unsigned char));
    cudaMalloc((void**)&d_mask, size_mask*sizeof(unsigned char));
    cudaMalloc((void**)&d_output,size_gpu_output*sizeof(unsigned char));

    cudaMemcpy(d_mapping, mapping, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, gpu_input, sizeof(unsigned char)* size_gpu_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, sizeof(unsigned char)*size_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output, sizeof(unsigned char)*size_gpu_output, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int numBlocks = (n+threadsPerBlock-1)/threadsPerBlock;
    int size_andres = GAP_SIZE_BYTES * 2 * size_mask + ROW_SIZE_BYTES + 1 + 1024;
    unfoldkernel<<<numBlocks,threadsPerBlock>>>(d_mapping, d_input, d_mask, d_output, n, size_andres, size_mask);
    cudaMemcpy(gpu_output, d_output, size_gpu_output*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    
    list<row> gpu_bm;
    convert_1dimarr_to_bitmat(gpu_output, gpu_bm, n, size_andres);	
	
	return 0;
}