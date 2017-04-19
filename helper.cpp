#include "bitmat.hpp"

unsigned int table_col_bytes = 5;
unsigned int gnum_subs, gnum_preds, gnum_objs, gnum_comm_so;

void print_bitmat(list<row> bm){
    int n=bm.size();
    //cout << "My size is " << n << endl;
    for(list<row>::iterator it=bm.begin(); it!=bm.end(); it++){
        cout <<"rowid: " <<  (*it).rowid << endl;
        unsigned char* data = (*it).data;
        unsigned rowsize=0;
        memcpy(&rowsize, data, ROW_SIZE_BYTES);
        cout <<"rowsize: "<< rowsize << endl;
        data += ROW_SIZE_BYTES;
        bool flag = data[0] & 0x01;
        if(flag)
        	cout << "flag: 0x01\n";
        else
        	cout << "flag: 0x00\n";
        unsigned char format = data[0] & 0x02;
        if(format==0x02)
        	cout << "format: 0x02\n";
        else
        	cout << "format: 0x00\n";

        data++;

        unsigned int total_cnt = (rowsize-1)/GAP_SIZE_BYTES;

        /*for(unsigned int j=0;j < total_cnt; j++){
            unsigned int tmpcnt;
            memcpy(&tmpcnt, data, GAP_SIZE_BYTES);
            cout << tmpcnt << " ";
            data += GAP_SIZE_BYTES;
        }
        cout << endl;*/
    }
}

unsigned int last_set_bit(unsigned char *in, unsigned int size)
{
	if (size == 0)
		return 0;

	unsigned int last_set_bit = 0;

	for (unsigned int i = 0; i < size; i++) {
		if (in[i] == 0xff) {
			last_set_bit = (i+1)*8;
		} else if (in[i] > 0x00) {
			for (unsigned short j = 0; j < 8; j++) {
				if (in[i] & (0x80 >> j)) {
					last_set_bit = (i*8)+j+1;
				}
			}
		}
	}

	return last_set_bit;

}
	
unsigned long get_offset(char *dumpfile, unsigned int bmnum)
{
	//Get the offset
	char tablefile[1024];
	sprintf(tablefile, "%s_table", dumpfile);
	unsigned long offset = 0;
/*#if MMAPFILES
	char *mmapfile = mmap_table[string(tablefile)];
	memcpy(&offset, &mmapfile[(bmnum-1)*table_col_bytes], table_col_bytes);
#else*/
	int fd = open(tablefile, O_RDONLY);
	if (fd < 0) {
		cout << "*** ERROR opening " << tablefile << endl;
		assert (0);
	}

	lseek(fd, (bmnum-1)*table_col_bytes, SEEK_CUR);
	unsigned char tablerow[table_col_bytes];
	read(fd, tablerow, table_col_bytes);
	memcpy(&offset, tablerow, table_col_bytes);
	close(fd);
/*#endif*/
	return offset;

}

unsigned int load_from_dump_file(char *fname_dump, unsigned int bmnum, BitMat *bitmat, bool readtcnt,
			bool readarray, unsigned char *maskarr, unsigned int maskarr_size, int maskarr_dim,
			char *fpos, int fd, bool fold_objdim)
{
	unsigned int size = 0;
	assert((readtcnt && readarray) || (!readtcnt && !readarray && ((fd == 0) ^ (NULL == fpos))));

	if (fd == 0) {
		fd = open(fname_dump, O_RDONLY | O_LARGEFILE);
		if (fd < 0) {
			printf("*** ERROR opening dump file %s\n", fname_dump);
			assert(0);
		}
		unsigned long offset = get_offset(fname_dump, bmnum);
		///////
		//DEBUG: remove later
		////////
		cout << "load_from_dump_file: offset is " << offset << endl;
		if (offset > 0) {
			lseek(fd, offset, SEEK_CUR);
		}
	}

	assert((fd == 0) ^ (NULL == fpos));

//	cout << "load_from_dump_file: offset " << offset << endl;


	if (readtcnt) {
		bitmat->num_triples = 0;
		read(fd, &bitmat->num_triples, sizeof(unsigned int));
		//cout << "num_triples: " << bitmat->num_triples << endl;
//		total_size += sizeof(unsigned int);
		if (bitmat->num_triples == 0) {
			cout << "load_from_dump_file: 0 results" << endl;
			return bitmat->num_triples;
		}
	}
	unsigned long s = 0;
	if (readarray) {
		//cout << "here1\n";
		
		if (bitmat->subfold == NULL) {
		//	cout << "here2\n";
			bitmat->subfold = (unsigned char *) malloc (bitmat->subject_bytes * sizeof (unsigned char));
			memset(bitmat->subfold, 0, bitmat->subject_bytes * sizeof (unsigned char));

		}
		if (bitmat->objfold == NULL) {
		//	cout << "here3\n";
			bitmat->objfold = (unsigned char *) malloc (bitmat->object_bytes * sizeof (unsigned char));
			memset(bitmat->objfold, 0, bitmat->object_bytes * sizeof (unsigned char));
		}
	/*	s += bitmat->subject_bytes;
		s += bitmat->object_bytes;*/
		read(fd, bitmat->subfold, bitmat->subject_bytes);
		read(fd, bitmat->objfold, bitmat->object_bytes);
	}

	unsigned int rownum = 1;
	
	if (maskarr == NULL || maskarr_dim == COLUMN) {
		//cout << "mask array is null\n" << ROW_SIZE_BYTES << endl;
		for (unsigned int i = 0; i < bitmat->subject_bytes; i++) {
			if (bitmat->subfold[i] == 0x00) {
				rownum += 8;
			} else {
				for (short j = 0; j < 8; j++) {
					if (!(bitmat->subfold[i] & (0x80 >> j))) {
						rownum++;
						continue;
					}
	
					read(fd, &size, ROW_SIZE_BYTES);
	//				s+= ROW_SIZE_BYTES;
					unsigned char *data = (unsigned char *) malloc (size + ROW_SIZE_BYTES);
					memcpy(data, &size, ROW_SIZE_BYTES);
	//				cout << size << endl;
					read(fd, data + ROW_SIZE_BYTES, size);
	//				s+=size;
					struct row r = {rownum, data};
					bitmat->bm.push_back(r);
					rownum++;
				}
			}
		}

	} 
	//cout << s << endl;
	return bitmat->num_triples;

}

unsigned long count_size_of_bitmat(BitMat *bitmat)
{
	unsigned long size = 0;

	if (bitmat->bm.size() > 0) {
		for (std::list<struct row>::iterator it = bitmat->bm.begin(); it != bitmat->bm.end(); it++) {
			unsigned int rowsize = 0;
			memcpy(&rowsize, (*it).data, ROW_SIZE_BYTES);
			//cout << rowsize << endl;
			size += rowsize + ROW_SIZE_BYTES + sizeof((*it).rowid);
		}
	} /*else if (bitmat->vbm.size() > 0) {
		for (vector<struct row>::iterator it = bitmat->vbm.begin(); it != bitmat->vbm.end(); it++) {
			unsigned int rowsize = 0;
			memcpy(&rowsize, (*it).data, ROW_SIZE_BYTES);
			size += rowsize + ROW_SIZE_BYTES + sizeof((*it).rowid);
		}

	}*/
	if (bitmat->subfold != NULL) {
		size += sizeof(bitmat->subject_bytes);
	}
//	if (bitmat->subfold_prev != NULL) {
//		size += sizeof(bitmat->subject_bytes);
//	}
	if (bitmat->objfold != NULL) {
		size += sizeof(bitmat->object_bytes);
	}
//	if (bitmat->objfold_prev != NULL) {
//		size += sizeof(bitmat->object_bytes);
//	}

	return size;
}


void dgap_uncompress(unsigned char *in, unsigned int insize, unsigned char *out, unsigned int outsize)
{
	unsigned int tmpcnt = 0, bitcnt = 0;
	unsigned int cnt = 0, total_cnt = 0, bitpos = 0;
	bool flag;

	total_cnt = (insize-1)/GAP_SIZE_BYTES;
	flag = in[0] & 0x01;
	unsigned char format = in[0] & 0x02;

	while (cnt < total_cnt) {

		memcpy(&tmpcnt, &in[cnt*GAP_SIZE_BYTES+1], GAP_SIZE_BYTES);
		if (format == 0x02) {
			out[(tmpcnt-1)/8] |= (0x80 >> ((tmpcnt-1) % 8));
		} else {
			if (flag) {
				for (bitpos = bitcnt; bitpos < bitcnt+tmpcnt; bitpos++) {
					out[bitpos/8] |= (0x80 >> (bitpos % 8));
				}
			}
		}
		cnt++;
		flag = !flag;
		bitcnt += tmpcnt;
	}
	if (format == 0x02) {
		assert((tmpcnt-1)/8 < outsize);
	} else {
		assert((bitcnt-1)/8 < outsize);
	}
}

void simple_fold(BitMat *bitmat, int ret_dimension, unsigned char *foldarr, unsigned int foldarr_size)
{
//	printf("Inside fold\n");
	memset(foldarr, 0, foldarr_size);

	if (ret_dimension == ROW) {
		if (bitmat->last_unfold == ROW) {
			assert(foldarr_size == bitmat->subject_bytes);
			memcpy(foldarr, bitmat->subfold, bitmat->subject_bytes);
		} else {
			for (std::list<struct row>::iterator it = bitmat->bm.begin(); it != bitmat->bm.end(); it++) {
				unsigned int rowbit = (*it).rowid - 1;
				assert(rowbit/8 < foldarr_size);
				foldarr[rowbit/8] |= (0x80 >> (rowbit%8));
			}
		}
///////////////////////////////////////////////////////////////////////////////////////

	} else if (ret_dimension == COLUMN) {
		//cout << "yo\n";
		if (bitmat->last_unfold == COLUMN) {
			assert(foldarr_size == bitmat->object_bytes);
			memcpy(foldarr, bitmat->objfold, bitmat->object_bytes);
		} else {
			for (std::list<struct row>::iterator it = bitmat->bm.begin(); it != bitmat->bm.end(); it++) {
				unsigned char *data = (*it).data;
				unsigned rowsize = 0;
				memcpy(&rowsize, data, ROW_SIZE_BYTES);
				//cout << "before dgap_uncompress\n";
		//		cout << rowsize << " " << foldarr_size << endl;
				dgap_uncompress(data + ROW_SIZE_BYTES, rowsize, foldarr, foldarr_size);
				//cout << "after dgap_uncompress\n";

			}
		}
	} else {
		cout << "simple_fold: **** ERROR unknown dimension " << ret_dimension << endl;
		assert(0);
	}
}

unsigned long long int get_sizeof_1dimarr(list<row> bm){
    unsigned long long int res = 0;
    
    for(list<row>::iterator it=bm.begin(); it!=bm.end(); it++){
        res += ROW_SIZE_BYTES;
        unsigned char* data = (*it).data;
        unsigned int temp;
        memcpy(&temp, data, ROW_SIZE_BYTES);
        cout <<"temp: "<<	 temp << endl;
        res += temp;
    }
    return res;
}

void init_bitmat(BitMat *bitmat, unsigned int snum, unsigned int pnum, unsigned int onum, unsigned int commsonum, int dimension)
{
//	bitmat->bm = (unsigned char **) malloc (snum * sizeof (unsigned char *));
//	memset(bitmat->bm, 0, snum * sizeof (unsigned char *));
	bitmat->bm.clear();
	bitmat->num_subs = snum;
	bitmat->num_preds = pnum;
	bitmat->num_objs = onum;
	bitmat->num_comm_so = commsonum;

//	row_size_bytes = bitmat->row_size_bytes;
//	gap_size_bytes = bitmat->gap_size_bytes;
	bitmat->dimension = dimension;

	bitmat->subject_bytes = (snum%8>0 ? snum/8+1 : snum/8);
	bitmat->predicate_bytes = (pnum%8>0 ? pnum/8+1 : pnum/8);
	bitmat->object_bytes = (onum%8>0 ? onum/8+1 : onum/8);
	bitmat->common_so_bytes = (commsonum%8>0 ? commsonum/8+1 : commsonum/8);

	bitmat->subfold = (unsigned char *) malloc (bitmat->subject_bytes * sizeof(unsigned char));
	memset(bitmat->subfold, 0, bitmat->subject_bytes * sizeof(unsigned char));
	bitmat->objfold = (unsigned char *) malloc (bitmat->object_bytes * sizeof(unsigned char));
	memset(bitmat->objfold, 0, bitmat->object_bytes * sizeof(unsigned char));
//	bitmat->subfold_prev = NULL;
//	bitmat->objfold_prev = NULL;
	bitmat->single_row = false;
	bitmat->num_triples = 0;

}

void convert_bitmat_to_1dimarr(list<row> bm, unsigned long long int int* mapping, int n, unsigned char* gpu_input){
    //cout << "hello\n";
    unsigned char* total_data = gpu_input;
    //int gap_size = sizeof(unsigned int);
    list<row>::iterator it = bm.begin();
    int j = 0;
    for(unsigned int i=0; i < n; i++){
    	assert(it!=bm.end());
        //cout << i << endl;
        unsigned int rowid = (*it).rowid - 1;

        if(rowid == i){
            mapping[i] = total_data - gpu_input; 
            unsigned char* data = (*it).data;
            unsigned size=0;
            memcpy(&size, data, ROW_SIZE_BYTES);
            size += ROW_SIZE_BYTES;
            memcpy(total_data, data, size);
            total_data += size;
            it++;
        }
        else
            mapping[i] = -1;
    }
}

