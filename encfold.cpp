#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <cstring>
#include <stdlib.h>

using namespace std;

int main(){
    ifstream in;
    in.open("enc_mat");
    int r;
    in >> r;
    vector<pair<bool,vector<int> > > encmat(r);
    for(int i = 0; i<r; i++){
        in >> encmat[i].first;
        int size;
        in >> size;
        vector<int> v(size);
        for(int j=0;j<size;j++){
            in >> v[j];
        }
        encmat[i].second = v;
        v.clear();
    }

    int masksize = (r%8>0 ? r/8+1 : r/8);

    unsigned char* mask = (unsigned char*)malloc(masksize*sizeof(unsigned char));
    memset(mask,0,masksize);
    for(int i=0;i<r;i++){
        bool flag = encmat[i].first;
        vector<int> v = encmat[i].second;
        int count = 0;
        for(int j=0; j<v.size(); j++){
            if(flag){
                for(int k=count;k<count+v[j];k++){
                    mask[k/8] |= (0x80 >> k%8);
                }
                count+=v[j];
                flag=0;
            }
            else{
                count += v[j];
                flag=1;
            }

        }
    }
    for(int i=0;i<masksize;i++)
        printf("%d\n",mask[i]);
    return 0;
}
