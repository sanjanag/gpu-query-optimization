#include <iostream>
#include <stdio.h>
#include "bitmat.hpp"

using namespace std;



int main(){
	cout << "Hello World\n";
	//cout << table_col_bytes << endl;
	BitMat* bitmat = new BitMat;
	char dumpfile[1024] = "/work/scratch/datasets/dbpedia/all_data_bitmats/dbpedia565m_pos_odump";

	init_bitmat(bitmat, gnum_subs, gnum_preds, gnum_objs, gnum_comm_so, POS_BITMAT);

	for(unsigned int i=1; i <= gnum_objs; i++){
		cout << "i: " << i <<  endl; 
		unsigned int triples =  load_from_dump_file(dumpfile, i, bitmat, true, true, NULL, 0, 0, NULL, 0, true);	
		cout << "size of  bitmat: " <<  count_size_of_bitmat(bitmat) << endl;
		simple_fold(bitmat, COLUMN, bitmat->objfold, bitmat->object_bytes);
		clear_rows(bitmat, true, true, false);
	}

	
	/*cout << triples << endl;
	cout << bitmat->num_subs << '\t' << bitmat->num_preds << '\t' << bitmat->num_objs << '\t' << bitmat->num_comm_so << endl;
	cout << bitmat->bm.size() << endl;*/
	/*cout << "size of  bitmat: " <<  count_size_of_bitmat(bitmat) << endl;
	simple_fold(bitmat, COLUMN, bitmat->objfold, bitmat->object_bytes);*/
	return 0;
}