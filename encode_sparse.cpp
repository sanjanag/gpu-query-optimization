#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <list>
#include <cstring>
#include <vector>
#include <utility>

using namespace std;

struct row {
	unsigned int rowid; //nodeid from the orig graph
	unsigned char *data;
	bool operator<(const struct row &a) const
	{
		return (this->rowid < a.rowid);
	}
};

void get_row_bytes(vector<pair<bool,vector<int> > > comp_mat, int n, unsigned char* row_bytes){
    for(int i=0; i<n; i++){
        int idx = i/8;
        int pos = i%8;

        if(comp_mat[i].first || comp_mat[i].second[0]<n){
            row_bytes[idx] |= (0x80 >> pos);
        }
    }
}

void get_bitmat(list<row>& bm, unsigned char* row_bytes, int size_row_bytes, vector<pair<bool,vector<int> > > comp_mat){

    int rownum=0;

    for(int i=0; i<size_row_bytes; i++){
        if(row_bytes[i]==0x00){
            rownum += 8;
        }
        else{
            for(int j=0;j<8;j++){
                if(row_bytes[i] & (0x800 >> j)){
                    bool start = comp_mat[rownum].first;
                    vector<int> comp_row = comp_mat[rownum].second;

                    unsigned char* data;
                    int gap_size = sizeof(unsigned int);

                    int size = comp_row.size()*gap_size+1;
                    memcpy(data, &size, gap_size);

                    unsigned char* curr = data+gap_size;

                    *curr = (start ? 0x80 : 0x00);
                    curr++;

                    for(int k=0;k<comp_row.size();k++){
                        memcpy(curr, comp_row[k], gap_size);
                        curr+=gap_size;
                    }

                    row r = {rownum,data};
                    bm.push_back(r);
                }
                rownum++;
            }
        }
    }
}

void compress_sparse_bitmat(bool* raw, vector<pair<bool,vector<unsigned int> > >& comp_mat, int n){
    for(int i=0;i<n;i++){
        bool flag = raw[i*n];
        comp_mat[i].first = flag;
        int count = 0;
        vector<unsigned int> v;
        for(int j=0;j<n;j++){
            if(raw[i*n+j]){
                if(flag)
                    count++;
                else{
                    v.push_back(count);
                    flag =1;
                    count = 1;
                }
            }
            else{
                if(flag){
                    v.push_back(count);
                    flag=0;
                    count = 1;
                }
                else
                    count++;
            }
        }
        comp_mat[i].second = v;
        v.clear();
    }
}

int main(){
    ifstream in;
    in.open("data.in");
    int n=10;
    bool raw[n*n];

    //read sparse matrix from raw file
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            in >> raw[i*n+j];
    }

    //compress sparse raw matrix
    vector<pair<bool,vector<unsigned int> > > comp_mat(n);
    compress_sparse_bitmat(raw,comp_mat,n);

    //create row_bytes array
    int size_row_bytes = (n%8>0 ? n/8+1 : n/8);
    unsigned char* row_bytes = (unsigned char*)malloc(size_row_bytes*sizeof(unsigned char));
    memset(row_bytes,0,size);
    get_row_bytes(comp_mat, n, row_bytes);



    //create bitmat
    list<row> bm;

    get_bitmat(bm, row_bytes,size_row_bytes, comp_mat);

    return 0;
}
