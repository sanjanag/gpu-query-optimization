#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void addkernel(int (*a)[5], int (*b)[5], int (*c)[5], int n){
    int i = threadIdx.x;
    int j = threadIdx.y;
    if(i<n && j<n){
        c[i][j]=a[i][j]+b[i][j];
        printf("%d",a[i][j]);
    }
}

int main(){
    int **a,**b,**c;
    int rows,cols;
    rows = cols = 5;

    a = (int**)malloc(rows*sizeof(int*));
    for(int i=0;i<rows;i++){
        a[i] = (int*)malloc(cols*sizeof(int));
    }
    b = (int**)malloc(rows*sizeof(int*));
    for(int i=0;i<rows;i++){
        b[i] = (int*)malloc(cols*sizeof(int));
    }
    c = (int**)malloc(rows*sizeof(int*));
    for(int i=0;i<rows;i++){
        c[i] = (int*)malloc(cols*sizeof(int));
    }

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            a[i][j]=i;
            b[i][j]=j;
        }
    }
/*
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            c[i][j]=a[i][j]+b[i][j];
        }
    }
*/

    int **dev_a, **dev_b, **dev_c;
    dev_a = (int**)malloc(rows*sizeof(int*));
    dev_b = (int**)malloc(rows*sizeof(int*));
    dev_c = (int**)malloc(rows*sizeof(int*));

    for(int i=0;i<rows;i++){
        cudaMalloc((void**)&dev_a[i], cols*sizeof(int));
        cudaMalloc((void**)&dev_b[i], cols*sizeof(int));
        cudaMalloc((void**)&dev_c[i], cols*sizeof(int));
    }

    for(int i=0;i<rows;i++){
        cudaMemcpy(dev_a[i], a[i],sizeof(int)*cols,cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b[i], b[i],sizeof(int)*cols,cudaMemcpyHostToDevice);
    }

    dim3 threadsPerBlock(16,16);

    addkernel<<<1,threadsPerBlock>>>(dev_a,dev_b,dev_c, rows);

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            printf("%d ",c[i][j]);
        }
        printf("\n");
    }
    for(int i=0;i<rows;i++){
        cudaMemcpy(dev_c[i], c[i],sizeof(int)*cols, cudaMemcpyDeviceToHost);
    }
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            printf("%d ",c[i][j]);
        }
        printf("\n");
    }
/*    for(int i=0;i<rows;i++){
        cudaMalloc((void**)&dev_a[i], cols*sizeof(int));
    }*/
    return 0;
}
