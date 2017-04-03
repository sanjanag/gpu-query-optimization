#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <list>
#include <cstring>
#include <vector>
#include <utility>
//#include <cuda.h>

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

void print_bitmat(vector<row> bm){
    int n=bm.size();
    for(vector<row>::iterator it=bm.begin(); it!=bm.end(); it++){
        cout << (*it).rowid << endl;
        unsigned char* data = (*it).data;
        int size;
        memcpy(&size,data,sizeof(unsigned int));
        cout << size << " ";
        data+=sizeof(unsigned int);
        if(data[0])
            cout << 1 << " ";
        else
            cout << 0 << " ";
        //printf("flag: %d\n",*data);
        data++;
        for(int j=0;j<((size-1)/sizeof(unsigned int)); j++){
            int tmp;
            memcpy(&tmp, data, sizeof(unsigned int));
            cout << tmp << " ";
            //printf("%d ",tmp);
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

/*__global__ void foldkernel(int* mapping, unsigned char* input, unsigned char* res, int n,int size_mask, int split){
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
*/

void get_mask(unsigned char* mask, bool* raw_mask, int size_mask, int n){
    for(int i=0;i<n;i++){
        if(raw_mask[i]){
            mask[i/8] |= (0x80 >> (i%8));
        }
    }
}

void unfold_bitmat(vector<row> bm, unsigned char* mask, int size_mask, vector<row>& ubm){
    int gap_size = sizeof(unsigned int);
    unsigned int andres_size = 0;
    unsigned char *andres = (unsigned char *) malloc (gap_size * 2 * size_mask + gap_size + 1 + 1024);
    
    //vector<row> ubm;
    
    for(vector<row>::iterator it = bm.begin(); it!= bm.end(); it++){
        unsigned int andres_size = 0;
        unsigned char *andres = (unsigned char *) malloc (gap_size * 2 * size_mask + gap_size + 1 + 1024);
        unsigned int rowid = (*it).rowid;
        unsigned char* data  = (*it).data;
        unsigned int rowsize = 0;
        memcpy(&rowsize, data, gap_size);
        data += gap_size;
        unsigned int cnt = 0, total_cnt = (rowsize-1)/gap_size, tmpcnt = 0, triplecnt = 0,
                    prev1_bit = 0, prev0_bit = 0, tmpval = 0;
        bool flag = data[0], begin = true, p1bit_set = false, p0bit_set = false;

        while(cnt < total_cnt){
            memcpy(&tmpcnt, &data[cnt*gap_size+1], gap_size);

            if(flag){
                for(unsigned int i = triplecnt; i < triplecnt + tmpcnt; i++){
                    if((mask[i/8] & (0x80 >> (i%8))) == 0x00){
                        if(begin){
                            begin = false;
                            andres[gap_size] = 0x00;
                            andres_size = gap_size + 1;
                            p0bit_set = true;
                        }
                        else{
                            if(!p0bit_set){
                                memcpy(&andres[andres_size], &i, gap_size);
                                andres_size += gap_size;
                                p0bit_set = true;
                            }
                            else if(prev0_bit != (i-1)){
                                tmpval = i - prev0_bit - 1;
                                memcpy(&andres[andres_size], &tmpval, gap_size);
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
                                memcpy(&andres[andres_size], &i, gap_size);
                                andres_size += gap_size;
                                p1bit_set = true;
                            }
                            else if(prev1_bit != (i-1)){
                                tmpval = i - prev1_bit - 1;
                                memcpy(&andres[andres_size], &tmpval, gap_size);
                                andres_size += gap_size;
                            }
                        }
                        prev1_bit = i;
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
                        memcpy(&andres[andres_size], &tmpval, gap_size);
                        andres_size += gap_size;
                        p0bit_set = true;
                    }
                    else if(prev0_bit != (triplecnt-1)){
                        tmpval = triplecnt -prev0_bit - 1;
                        memcpy(&andres[andres_size], &tmpval, gap_size);
                        andres_size += gap_size;
                    }
                }
                prev0_bit = triplecnt + tmpcnt - 1;
            }

            
            flag = !flag;
            cnt++;
            triplecnt += tmpcnt;
        }

        if(prev1_bit > prev0_bit){
            if(!p0bit_set){
                tmpval = prev1_bit + 1;
                memcpy(&andres[andres_size], &tmpval, gap_size);
            }
            else{
                tmpval = prev1_bit - prev0_bit;
                memcpy(&andres[andres_size], &tmpval, gap_size);
            }
            andres_size +=  gap_size;
        }
        else{
            if(!p1bit_set){
                /*data -= gap_size;
                free(data);
                it = bm.erase(it);*/
                continue;
            }
            else{
                tmpval = prev0_bit - prev1_bit;
                memcpy(&andres[andres_size], &tmpval, gap_size);
            }
            andres_size += gap_size;
        }

        tmpval = andres_size - gap_size;
        memcpy(andres, &tmpval, gap_size);
        row r = {rowid,andres};
        ubm.push_back(r);
        /*data -= gap_size;
        free(data);
        (*it).data = (unsigned char*)malloc(andres_size);
        memcpy((*it).data, andres, andres_size);*/
    }
}

void print_raw_mask(bool* mask, int n){
    for(int i=0;i<n;i++)
        cout << mask[i] << " ";
    cout << endl;
}

void brute_force_unfold(bool* in, bool* mask, bool* out, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            out[i*n+j] = in[i*n+j] & mask[j];
        }
    }
}

void test_unfold(vector<row> corr, vector<row> sample){
    int m = corr.size();
    int n = sample.size();
    if(m!=n){
        cout << "FAIL1\n";
        return;
    }

    for(int i=0; i<n; i++){
        unsigned int rowid1 = corr[i].rowid, rowid2 = sample[i].rowid;
        if(rowid1!=rowid2){
            cout << "FAIL2\n";
            return;       
        }

        unsigned char* data1 = corr[i].data;
        unsigned char* data2 = sample[i].data;
        
        int size1, size2;
        memcpy(&size1, data1, sizeof(unsigned int));
        memcpy(&size2, data2, sizeof(unsigned int));
        if(size1!=size2){
            cout << "FAIL3\n";
            return;       
        }
        
        
        data1 += sizeof(unsigned int);
        data2 += sizeof(unsigned int);        


        if(data1[0] ^ data2[0]){
            cout <<i <<  "FAIL4\n";
            return;
        }

        data1++;
        data2++;

        for(int j=0; j<((size1-1)/sizeof(unsigned int)); j++){
            int tmp1, tmp2;
            memcpy(&tmp1, data1, sizeof(unsigned int));
            memcpy(&tmp2, data2, sizeof(unsigned int));
            if(tmp1!=tmp2){
                cout << "FAIL5\n";
                return;                 
            }
            
            data1 += sizeof(unsigned int);
            data2 += sizeof(unsigned int);
        }
    }   

    cout << "PASS\n";
}

int get_size_gpu_output(unsigned char* mask, int n){
    int gap_size = sizeof(unsigned int);
    int out_size = gap_size+1;
    bool flag = mask[0] & 0x80;
    int count = 0;
    vector<int> v;
    for(int j=0; j<n; j++){
        if(mask[j/8] & (0x80 >> (j%8))){
            if(flag)
                count++;
            else{
                v.push_back(count);
                flag = 1;
                count = 1;
            }
        }
        else{
            if(flag){
                v.push_back(count);
                flag=0;
                count = 1;
            }
            else{
                count++;
            }
        }
    }
    v.push_back(count);
    out_size += (gap_size*v.size());    
    return out_size;
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

    bool raw_mask[n];

    for(int i=0;i<n;i++)
        in >> raw_mask[i];

    bool raw_out[n*n];
    brute_force_unfold(raw, raw_mask, raw_out, n);
    vector<pair<bool,vector<unsigned int> > > corr_out(n);
    compress_sparse_bitmat(raw_out, corr_out, n);
    int size_row_bytes = (n%8>0 ? n/8+1 : n/8);
    unsigned char* out_row_bytes = (unsigned char*)malloc(size_row_bytes*sizeof(unsigned char));
    memset(out_row_bytes, 0, size_row_bytes);
    get_row_bytes(corr_out, n, out_row_bytes);
    vector<row> out_bm;
    get_bitmat(out_bm, out_row_bytes, size_row_bytes, corr_out);
    
    //CONVERT RAW INPUT TO BITMAT
    //compress sparse raw matrix
    vector<pair<bool,vector<unsigned int> > > comp_mat(n);
    compress_sparse_bitmat(raw,comp_mat,n);

    //create row_bytes array
    //int size_row_bytes = (n%8>0 ? n/8+1 : n/8);
    unsigned char* row_bytes = (unsigned char*)malloc(size_row_bytes*sizeof(unsigned char));
    memset(row_bytes,0,size_row_bytes);
    get_row_bytes(comp_mat, n, row_bytes);

    //create bitmat
    vector<row> bm;
    get_bitmat(bm, row_bytes,size_row_bytes, comp_mat);
    

    //CONVERT RAW MASK TO REQUIRED FORMAT
    int size_mask = (n%8>0 ? n/8+1 : n/8);
    unsigned char* mask = (unsigned char*)malloc(sizeof(unsigned char)*size_mask);
    memset(mask,0,size_mask);
    get_mask(mask, raw_mask, size_mask, n);
    
    //CPU UNFOLD
    vector<row> ubm;
    unfold_bitmat(bm, mask, size_mask, ubm);
    
    //TESTER FUNCTION
    test_unfold(out_bm, ubm);
    
    int mapping[n];
    for(int i=0; i<n; i++)
        mapping[i] = i;

    int size_gpu_input = get_sizeof_1dimarr(bm);
    unsigned char* gpu_input = (unsigned char*)malloc(size_gpu_input*sizeof(unsigned char));
    convert_bitmat_to_1dimarr(bm, mapping, n, gpu_input);

    int size_gpu_output =  get_size_gpu_output(mask, n);
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
    print_mask(gpu_mask,size_mask);*/

    return 0;
}
