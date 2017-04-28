#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

int main(int argc, char* argv[]){
    int n = atoi(argv[1]);
    thrust::host_vector<int> H(n)
    return 0;
}

