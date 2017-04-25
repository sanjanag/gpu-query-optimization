#include <iostream>
#include <stdio.h>
#include "bitmat.hpp"
#include <sys/time.h>

using namespace std;


/*void convert_bitmat_to_gpu_input(BitMat* bitmat, unsigned char* gpu_input, long long int* mapping, unsigned int num_subs){
	unsigned char* curr_row = gpu_input;

	if(bitmat->bm.size() > 0){
		for(list<struct row>::iterator it = bitmat->bm.begin(); it != bitmat->bm.end(); it++){
			unsigned int rowbit = (*it).rowid -1;
			mapping[rowbit] = curr_row - gpu_input;
			unsigned int rowsize = 0;
			memcpy(&rowsize, (*it).data, ROW_SIZE_BYTES);
			memcpy(curr_row, (*it).data, rowsize + ROW_SIZE_BYTES);
			curr_row += (rowsize+ROW_SIZE_BYTES);
		}
	}
}*/

int main(){
	cout << "Hello World\n";
	BitMat* bitmat = new BitMat;
	char dumpfile[1024] = "/work/scratch/datasets/dbpedia/all_data_bitmats/dbpedia565m_spo_pdump";

	init_bitmat(bitmat, gnum_subs, gnum_preds, gnum_objs, gnum_comm_so, SPO_BITMAT);

	unsigned int triples =  load_from_dump_file(dumpfile, 1, bitmat, true, true, NULL, 0, 0, NULL, 0, true);
	
	cout << "1" << " " << bitmat->bm.size() << endl;	
	
	
	struct timeval t1, t2;
    double elapsedTime;
    
    gettimeofday(&t1, NULL);

	simple_fold(bitmat, COLUMN, bitmat->objfold, bitmat->object_bytes);
	
	gettimeofday(&t2, NULL);
    
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    cout << elapsedTime << " ms.\n";	
	

	long long int mapping[gnum_subs];
	for(unsigned int i=0; i < gnum_subs; i++){
		mapping[i] = -1;
	}

	unsigned long long int gpu_input_size = get_size_of_gpu_input(bitmat);
	cout << gpu_input_size << endl;

	unsigned char* gpu_input = (unsigned char*)malloc(gpu_input_size * sizeof(unsigned char));

	convert_bitmat_to_gpu_input(bitmat, gpu_input, mapping, gnum_subs);

	return 0;
}