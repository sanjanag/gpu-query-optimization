#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <fstream>
#include <utility>


using namespace std;


int main(){
    ifstream in;
    in.open("bitmat.out");

    int r,c;
    in >> r >> c;
    int n=r;
    unsigned char* bitmat = (unsigned char*)malloc(sizeof(char)*r*c);

    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            int tmp;
            in >> tmp;
            bitmat[i*c+j] = tmp;
        }
    }


    vector<pair<bool,vector<int> > > enc_mat(r);

    for(int i=0; i<r ;i++){
        bool flag = 0x80 & bitmat[i*c];
        enc_mat[i].first = flag;
        vector<int> v;

        int count = 0;
        for(int j=0; j<n; j++){
            int idx = i*c+j/8;
            if(bitmat[idx] & (0x80 >> (j%8))){
                if(flag){
                    count++;
                }
                else{
                    v.push_back(count);
                    flag = 1;
                    count = 1;
                }
            }
            else{
                if(!flag)
                    count++;
                else{
                    v.push_back(count);
                    flag=0;
                    count = 1;
                }
            }

        }
        v.push_back(count);
        enc_mat[i].second = v;
        v.clear();
    }
    ofstream out;
    out.open("enc_mat");
    out << r << endl;
    for(int i=0;i<r;i++){
        out << enc_mat[i].first << endl;
        for(int j=0;j<enc_mat[i].second.size();j++){
            out << enc_mat[i].second[j] << '\t';
        }
        out << endl;
    }
    out.close();


    return 0;
}
