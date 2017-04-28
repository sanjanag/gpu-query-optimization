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

__device__ unsigned long gpu_count_bits_in_row(unsigned char *in, unsigned int size)
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

__global__ void unfoldkernel(int* mapping, unsigned char* input, unsigned char* mask, unsigned char* output, unsigned int num_subs, unsigned int max_andres_size, unsigned int mask_size, unsigned int num_objs
	){
    
    unsigned int gap_size = 4, row_size = 4; 
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(row < num_subs){
        if(mapping[row] == -1)
            return;

        unsigned int andres_size = 0;
        unsigned int maskarr_bits = gpu_count_bits_in_row(mask, mask_size);

        if (maskarr_bits == num_objs) {
            return;
        }

        if(mask_size == 0){
            return;
        }

        unsigned char* andres = output + max_andres_size * row; //grave mistake here
        unsigned char* data  = input+mapping[row];
        unsigned int rowsize = 0;
        Memcpy(&rowsize, data, row_size);
        data += row_size;

        unsigned int cnt = 0, total_cnt = (rowsize-1)/gap_size, tmpcnt = 0, triplecnt = 0,
                    prev1_bit = 0, prev0_bit = 0, tmpval = 0;
        bool flag = data[0] & 0x01, begin = true, p1bit_set = false, p0bit_set = false;
        unsigned char format = data[0] & 0x02;

        if(format == 0x02){
            begin = false;
            andres[row_size] = 0x02;
            andres_size = row_size + 1;
        }

        while(cnt < total_cnt){

            Memcpy(&tmpcnt, &data[cnt*gap_size+1], gap_size);

            if(format==0x02){
                if((tmpcnt-1)/8 >= mask_size){
                    //pass
                }
                else if(mask[(tmpcnt-1)/8] & (0x80 >> ((tmpcnt-1)%8))){
                    Memcpy(&andres[andres_size], &tmpcnt, gap_size);
                    andres_size += gap_size;
                }
                else{
                	//pass
                }
            }
            else if (format == 0x00){
                if(flag){
                    if (triplecnt/8 >= mask_size) { //yet to understand
                        prev0_bit = triplecnt + tmpcnt - 1;
                	}
                    else{
                        for(unsigned int i = triplecnt; i < triplecnt + tmpcnt; i++){
                            if(i/8 >= mask_size){
                                
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
                                    andres_size = row_size + 1;
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
                                    andres[gap_size] = 0x01;
                                    andres_size = row_size + 1;
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
                        andres[row_size] = 0x00;
                        andres_size = row_size+1;
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
            else if(prev1_bit == prev0_bit && p1bit_set && p0bit_set){
            	assert(0);
            }
            if((!andres[row_size]) && andres_size == 1+row_size){
            	//pass
            }
        }
        

        tmpval = andres_size - row_size;
        Memcpy(andres, &tmpval, row_size);
    }    
}


/*void test_unfold(list<row> corr, list<row> sample){

    int m = corr.size();
    int n = sample.size();
    
    if(m!=n){
        cout << "FAIL1\n";
        return;
    }

    for(list<struct row>::iterator corr_it = corr.begin(), sample_it = sample.begin(); corr_it != corr.end(), sample_it != sample.end(); corr_it++,sample_it++){
        unsigned int rowid1 = corr[i].rowid, rowid2 = sample[i].rowid;
        if(rowid1!=rowid2){
            cout << "FAIL2\n";
            return;       
        }

        unsigned char* data1 = corr[i].data;
        unsigned char* data2 = sample[i].data;
        
        unsigned int size1, size2; 
        memcpy(&size1, data1, sizeof(unsigned int));
        memcpy(&size2, data2, sizeof(unsigned int));
        if(size1!=size2){
            cout << "FAIL3\n";
            return;       
        }
        
        
        data1 += ROW_SIZE_BYTES;
        data2 += ROW_SIZE_BYTES;        


        if(data1[0] != data2[0]){
            cout << i <<  "FAIL4\n";
            return;
        }

        data1++;
        data2++;

        for(unsigned int j=0; j<((size1-1)/GAP_SIZE_BYTES); j++){
            unsigned int tmp1, tmp2;
            memcpy(&tmp1, data1, GAP_SIZE_BYTES);
            memcpy(&tmp2, data2, GAP_SIZE_BYTES);
            if(tmp1!=tmp2){
                cout << "FAIL5\n";
                return;                 
            }
            
            data1 += GAP_SIZE_BYTES;
            data2 += GAP_SIZE_BYTES;
        }
    }   

    cout << "PASS\n";
}*/


list<row> convert_gpu_output_to_bitmat(unsigned char* output, unsigned int num_subs, unsigned int andres_size){
    cout << num_subs << " " << andres_size << endl;
    list<row> bm;
    for(unsigned long long int i=0; i<num_subs; i++){
    	//cout << i << endl;
        unsigned char* data = output + andres_size*i;
        unsigned int rowsize = 0;
        //cout << "s3\n" << data[0] << endl;
        memcpy(&rowsize, data, ROW_SIZE_BYTES);
        //cout << "s1\n";
        unsigned int total_cnt = (rowsize-1)/GAP_SIZE_BYTES;

        if(total_cnt == 0){
            continue;
        }

        unsigned char* rdata = (unsigned char*)malloc(ROW_SIZE_BYTES + 1 + total_cnt*GAP_SIZE_BYTES);
        memcpy(rdata, data, (ROW_SIZE_BYTES + 1 + total_cnt*GAP_SIZE_BYTES));
        //cout << "s2\n";
        row r = {i+1,rdata};
        bm.push_back(r);
    }
   return bm;
}

int main(int argc, char* argv[]){
	cout << "Hello World!\n";

	BitMat* bitmat = new BitMat;
	char dumpfile[1024] = "/data/gpuuser1/gpu_query_opt/dump/dbpedia565m_spo_pdump";

	init_bitmat(bitmat, gnum_subs, gnum_preds, gnum_objs, gnum_comm_so, SPO_BITMAT);
	unsigned int node = atoi(argv[1]);
	
	unsigned int triples =  load_from_dump_file(dumpfile, node, bitmat, true, true, NULL, 0, 0, NULL, 0, true);
	
	cout << node << " " << bitmat->bm.size() << endl;	
	
	
	
	simple_fold(bitmat, COLUMN, bitmat->objfold, bitmat->object_bytes);

	unsigned int mask_size = bitmat->object_bytes;
	unsigned char* maskarr = (unsigned char*)malloc(mask_size * sizeof(unsigned char));
	memcpy(maskarr, bitmat->objfold, mask_size);
	//simple_unfold(bitmat, maskarr, mask_size, COLUMN, 3);	
	//cout << "cpu_unfold\n";
	
    int* mapping = new int[gnum_subs];


	for(unsigned int i=0; i < gnum_subs; i++){
		mapping[i] = -1;
	}

	unsigned int andres_size = GAP_SIZE_BYTES * 2 * mask_size + ROW_SIZE_BYTES + 1 + 1024;
	unsigned long long int  gpu_output_size = andres_size * gnum_subs;
	unsigned char *d_input, *d_maskarr, *d_output;
	int* d_mapping;
	
	int threadsPerBlock = 512;
    int numBlocks = (gnum_subs%threadsPerBlock > 0 ? gnum_subs/threadsPerBlock + 1 : gnum_subs/threadsPerBlock);

	unsigned long long int gpu_input_size = get_size_of_gpu_input(bitmat);
	unsigned char* gpu_input = (unsigned char*)malloc(gpu_input_size * sizeof(unsigned char));
	unsigned char* gpu_output = (unsigned char*)malloc(gpu_output_size * sizeof(unsigned char));
	memset(gpu_output, 0, gpu_output_size);


	convert_bitmat_to_gpu_input(bitmat, gpu_input, mapping, gnum_subs);
	//cout << "here1\n";
	cudaMalloc((void**)&d_mapping, gnum_subs * sizeof(int));
    cudaMalloc((void**)&d_input, gpu_input_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_maskarr, mask_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, gpu_output_size * sizeof(unsigned char));

    cudaMemcpy(d_mapping, mapping, gnum_subs * sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, gpu_input, gpu_input_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maskarr, maskarr, mask_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, gpu_output, gpu_output_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    unfoldkernel<<<numBlocks,threadsPerBlock>>>(d_mapping, d_input, d_maskarr, d_output, gnum_subs, andres_size, mask_size, bitmat->num_objs);
    
    cudaMemcpy(gpu_output, d_output, gpu_output_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //cout << "here2\n";
    list<row> gpu_bm;

    gpu_bm =  convert_gpu_output_to_bitmat(gpu_output, gnum_subs, andres_size);	
    cout << "here3\n";
    simple_unfold(bitmat, maskarr, mask_size, COLUMN, 3);
 //   test_unfold(bitmat->bm, gpu_bm);

	return 0;
}