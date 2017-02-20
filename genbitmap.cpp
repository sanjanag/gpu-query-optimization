#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>

using namespace std;


void print(bool* a, int m, int n){
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++)
			cout << a[i*n+j] << '\t';
		cout << endl;
	}
	cout << endl;
}

int main(){

    ifstream in;
    in.open("data1e8.in");
    FILE* fp;
    fp = fopen("bitmat.out","w+");

    int n = 9;
    bool* inp = (bool*)malloc(sizeof(bool)*n*n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            in >> inp[i*n+j];
    }

    int cols = (n%8>0 ? n/8+1 : n/8);
    int bitmat_size = n*cols;
    unsigned char* bitmat = (unsigned char*)malloc(sizeof(unsigned char)*bitmat_size);

    memset(bitmat,0,bitmat_size);
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            int idx = i*cols+j/8;
            int pos= 0x80 >> (j%8);
            if(inp[i*n+j] == 1)
                bitmat[i*cols+j/8] |= pos;
        }
    }
    fprintf(fp, "%d %d\n",n,cols);
    for(int i=0;i<n;i++){
        for(int j=0;j<cols;j++)
            fprintf(fp,"%d ",bitmat[i*cols+j]);
        fprintf(fp,"\n");
    }
    fclose(fp);
    in.close();
    return 0;
}
