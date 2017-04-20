#ifndef _HELPER_H_
#define _HELPER_H_

#include <iostream>
#include <vector>
#include <map>
#include <utility>
//#include <unordered_set>
#include <stack>
#include <algorithm>
#include <fstream>
#include <list>
#include <set>
#include <cstring>
#include <string>
#include <cmath>
#include <fcntl.h>
#include <stdio.h>
//#include <math.h>
#include <sys/mman.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <time.h>

using namespace std;	 

#define ROW					10012
#define COLUMN				10013
#define ROW_SIZE_BYTES		4
#define GAP_SIZE_BYTES		4
#define POS_BITMAT			7

extern unsigned int table_col_bytes;
extern unsigned int gnum_subs, gnum_preds, gnum_objs, gnum_comm_so;

struct row {
	unsigned int rowid; //nodeid from the orig graph
	unsigned char *data;
	bool operator<(const struct row &a) const
	{
		return (this->rowid < a.rowid);
	}
};



typedef struct BitMat {
	list<struct row> bm;
	vector<struct row> vbm;
	bool single_row;
	unsigned int num_subs, num_preds, num_objs, num_comm_so; //num_rows, num_totalBMs, num_columns, num_comm_so
//	unsigned int row_size_bytes, gap_size_bytes;
	unsigned int subject_bytes, predicate_bytes, object_bytes, common_so_bytes, dimension, last_unfold;
	unsigned long num_triples;
	unsigned char *subfold; //row_bytearr
	unsigned char *objfold; //column_bytearr
//	unsigned char *subfold_prev;
//	unsigned char *objfold_prev;

	void freebm(void)
	{
		for (std::list<struct row>::iterator it = bm.begin(); it != bm.end(); ){
			free((*it).data);
			it = bm.erase(it);
		}
		if (subfold != NULL) {
			free(subfold);
			subfold = NULL;
		}
		if (objfold != NULL) {
			free(objfold);
			objfold = NULL;
		}
		num_triples = 0;
	}

//	void freerows(void)
//	{
//		for (std::list<struct row>::iterator it = bm.begin(); it != bm.end(); ){
//			free((*it).data);
//			it = bm.erase(it);
//		}
//	}

	void reset(void)
	{
		for (std::list<struct row>::iterator it = bm.begin(); it != bm.end(); ){
			free((*it).data);
			it = bm.erase(it);
		}
		if (subfold != NULL) {
			memset(subfold, 0, subject_bytes);
		}
		if (objfold != NULL) {
			memset(objfold, 0, object_bytes);
		}
		num_triples = 0;
	}

	BitMat()
	{
		num_subs = num_preds = num_objs = num_comm_so = num_triples = 0;
		subfold = objfold = NULL;
	}

	~BitMat()
	{
		for (std::list<struct row>::iterator it = bm.begin(); it != bm.end(); ){
			free((*it).data);
			it = bm.erase(it);
		}
		if (subfold != NULL)
			free(subfold);
		if (objfold != NULL)
			free(objfold);
//		free(subfold_prev);
//		free(objfold_prev);
	}

} BitMat;

unsigned int load_from_dump_file(char *fname_dump, unsigned int bmnum, BitMat *bitmat,
		bool readtcnt, bool readarray, unsigned char *maskarr, unsigned int mask_size, int maskarr_dim,
		char *fpos, int fd, bool fold_objdim);

unsigned long get_offset(char *fname, unsigned int bmnum);
void simple_fold(BitMat *bitmat, int ret_dimension, unsigned char *foldarr, unsigned int size);
void dgap_uncompress(unsigned char *in, unsigned int insize, unsigned char *out, unsigned int outsize);
void print_bitmat(list<row> bm);
void init_bitmat(BitMat *bitmat, unsigned int snum, unsigned int pnum, unsigned int onum, unsigned int commsonum, int dimension);
unsigned int count_subject_bytes(unsigned char* data);
unsigned long count_size_of_bitmat(BitMat *bitmat);
unsigned long long int get_sizeof_1dimarr(list<row> bm);
void convert_1dimarr_to_bitmat(unsigned char* input, list<row>& bm, int n, int size_andres);
void convert_bitmat_to_1dimarr(list<row> bm,int* mapping, int n, unsigned char* gpu_input);
#endif