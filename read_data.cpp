#include <iostream>
#include <stdio.h>
#include "bitmat.hpp"

using namespace std;



int main(){
	cout << "Hello World\n";
	BitMat* bitmat = new BitMat;
	char dumpfile[1024] = "/work/scratch/datasets/dbpedia/all_data_bitmats/dbpedia565m_spo_pdump";

	init_bitmat(bitmat, gnum_subs, gnum_preds, gnum_objs, gnum_comm_so, SPO_BITMAT);

	unsigned int triples =  load_from_dump_file(dumpfile, 1, bitmat, true, true, NULL, 0, 0, NULL, 0, true);
	
	cout << "1" << " " << bitmat->bm.size() << endl;	
	
	simple_fold(bitmat, COLUMN, bitmat->objfold, bitmat->object_bytes);
	
	return 0;
}