#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

bool check_ones(bool* mat, int row, int n){
    for(int i=0;i<n;i++){
        if(mat[row*n+i])
            return true;
    }
    return false;
}

int main(){
    ifstream in;
    in.open("data.in");
    int n=10;
    bool raw[n*n];

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            in >> raw[i*n+j];
    }

    int size_row_bytes = (n%8>0 ? n/8+1 : n/8);
    unsigned char* row_bytes = (unsigned char*)malloc(size_row_bytes*sizeof(unsigned char));
    memset(row_bytes,0,size_row_bytes);

    for(int i=0; i<n; i++){
        int idx = i/8;
        int pos = i%8;

        if(check_ones(raw,i,n)){
            row_bytes[idx] |= (0x80 >> pos);
        }
    }


    return 0;
}
