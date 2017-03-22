#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <list>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda.h>

using namespace std;

struct row {
	unsigned int rowid; //nodeid from the orig graph
	unsigned char *data;
	bool operator<(const struct row &a) const
	{
		return (this->rowid < a.rowid);
	}
};

void get_row_bytes(vector<pair<bool,vector<unsigned int> > > comp_mat, int n, unsigned char* row_bytes){
    for(int i=0; i<n; i++){
        int idx = i/8;
        int pos = i%8;

        if(comp_mat[i].first || comp_mat[i].second[0]<n){
            row_bytes[idx] |= (0x80 >> pos);
        }
    }
}

void get_bitmat(vector<row>& bm, unsigned char* row_bytes, int size_row_bytes, vector<pair<bool,vector<unsigned int> > > comp_mat){
    int rownum=0;

    for(int i=0; i<size_row_bytes; i++){
        if(row_bytes[i]==0x00){
            rownum += 8;
        }
        else{
            for(int j=0;j<8;j++){
                if(row_bytes[i] & (0x80 >> j)){
                    bool start = comp_mat[rownum].first;
                    vector<unsigned int> comp_row = comp_mat[rownum].second;

                    int gap_size = sizeof(unsigned int);
                    int size = comp_row.size()*gap_size+1;
                    unsigned char* data = (unsigned char*)malloc(size+gap_size);

                    memcpy(data, &size, gap_size);

                    unsigned char* curr = data+gap_size;

                    *curr = (start ? 0x80 : 0x00);
                    curr++;

                    for(int k=0;k<comp_row.size();k++){
                        unsigned int tmp = comp_row[k];
                        memcpy(curr, &tmp, gap_size);
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
        v.push_back(count);
        comp_mat[i].second = v;
        v.clear();
    }
}

void print(bool* mat, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            printf("%d ",mat[i*n+j]);
        cout << endl;
    }
}

void print_comp_mat(vector<pair<bool,vector<unsigned int> > > comp_mat, int n){
    for(int i=0;i<n;i++){
        cout << comp_mat[i].first << endl;
        for(int j=0;j<comp_mat[i].second.size();j++)
            printf("%d ",comp_mat[i].second[j]);
        cout << endl;
    }
}

void print_bitmat(list<row> bm){
    int n=bm.size();
    for(list<row>::iterator it=bm.begin(); it!=bm.end(); it++){
        cout <<"rowid: " <<  (*it).rowid << endl;
        unsigned char* data = (*it).data;
        int size;
        memcpy(&size,data,sizeof(unsigned int));
        cout <<"size: "<< size << endl;
        data+=sizeof(unsigned int);
        printf("flag: %d\n",*data);
        data++;
        for(int j=0;j<((size-1)/sizeof(unsigned int)); j++){
            int tmp;
            memcpy(&tmp, data, sizeof(unsigned int));
            printf("%d ",tmp);
            data+=sizeof(unsigned int);
        }
        cout << endl;
    }
}

void print_mask(unsigned char* mask, int size){
    for(int i=0;i<size;i++)
        printf("%d ",mask[i]);
    cout << endl;
}

void fold_bitmat(vector<row> bm, unsigned char* mask, int size_mask){
    //print_mask(mask,size_mask);
    int n = bm.size();
    int gap_size = sizeof(unsigned int);
    for(vector<row>::iterator it=bm.begin(); it!=bm.end(); it++){
        unsigned char* data=(*it).data;
        int size;
        memcpy(&size, data, gap_size);
        int nums = (size-1)/gap_size;
        int count=0;
        data+=gap_size;
        bool flag = *data;
        data++;
        for(int i=0;i<nums;i++){
            int tmp;
            memcpy(&tmp, data, gap_size);
            data+=gap_size;
            if(flag){
                for(int pos=count;pos <(count+tmp);pos++){
                    mask[pos/8] |= (0x80 >> (pos%8));
                }
            }
            count+=tmp;
            flag = !flag;
        }
        //print_mask(mask,size_mask);
    }
}

int get_sizeof_1dimarr(vector<row> bm){
    int res = 0;
    int gap_size = sizeof(unsigned int);
    for(int i=0; i<bm.size(); i++){
        res += gap_size;
        unsigned char* data = bm[i].data;
        int temp;
        memcpy(&temp, data, gap_size);
        res += temp;
    }
    return res;
}

void convert_bitmat_to_1dimarr(vector<row> bm, int* mapping, int n, unsigned char* gpu_input){
    //cout << "hello\n";
    unsigned char* total_data = gpu_input;
    int gap_size = sizeof(unsigned int);
    int j = 0;
    for(int i=0; i < n; i++){
        //cout << i << endl;
        int rowid = bm[j].rowid;

        if(rowid == i){
            mapping[i] = total_data - gpu_input; 
            unsigned char* data = bm[j].data;
            int size;
            memcpy(&size, data, gap_size);
            size += gap_size;
            memcpy(total_data, data, size);
            total_data += size;
            j++;
        }
        else
            mapping[i] = -1;
    }
}

__global__ void foldkernel(int* mapping, unsigned char* input, unsigned char* res, int n,int size_mask, int split){
    int threadid = blockIdx.x*blockDim.x+ threadIdx.x;
    int i = threadid*split;
    int gap_size = sizeof(unsigned int);
    
    for(int k = 0; k < split; k++){
        int rowid = i+k;
        if(rowid >= n)
            break;
        if(mapping[rowid]==-1)
            continue;
        unsigned char* data = input+mapping[rowid];
        int size;
        memcpy(&size, data, gap_size);
        int nums = (size-1)/gap_size;
        int count=0;
        data+=gap_size;
        bool flag = *data;
        data++;
        for(int i=0;i<nums;i++){
            int tmp;
            memcpy(&tmp, data, gap_size);
            data+=gap_size;
            if(flag){
                for(int pos=count;pos <(count+tmp);pos++){
                    res[threadid*size_mask +  pos/8] |= (0x80 >> (pos%8));
                }
            }
            count+=tmp;
            flag = !flag;
        }
    }
}

int main(int argc, char* argv[]){
    ifstream in;
    in.open("data.in");
    int n=atoi(argv[1]);
    bool raw[n*n];

    //read sparse matrix from raw file
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            in >> raw[i*n+j];
    }
    //print(raw,n);
    //compress sparse raw matrix
    vector<pair<bool,vector<unsigned int> > > comp_mat(n);
    compress_sparse_bitmat(raw,comp_mat,n);

    //create row_bytes array
    int size_row_bytes = (n%8>0 ? n/8+1 : n/8);
    unsigned char* row_bytes = (unsigned char*)malloc(size_row_bytes*sizeof(unsigned char));
    memset(row_bytes,0,size_row_bytes);
    get_row_bytes(comp_mat, n, row_bytes);

    //create bitmat
    vector<row> bm;
    get_bitmat(bm, row_bytes,size_row_bytes, comp_mat);
    //print_bitmat(bm);
    
    int size_mask = (n%8>0 ? n/8+1 : n/8);
    unsigned char* mask = (unsigned char*)malloc(sizeof(unsigned char)*size_mask);
    memset(mask,0,size_mask);
    fold_bitmat(bm, mask, size_mask);
    print_mask(mask,size_mask);

    int mapping[n];
    for(int i=0;i<n;i++)
        mapping[i] = i;

    int size_gpu_input = get_sizeof_1dimarr(bm);
    unsigned char* gpu_input = (unsigned char*)malloc(size_gpu_input*sizeof(unsigned char));
    //cout << "done\n";
    convert_bitmat_to_1dimarr(bm, mapping, n, gpu_input);
    //cout << "done\n";
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
    print_mask(gpu_mask,size_mask);

    return 0;
}
