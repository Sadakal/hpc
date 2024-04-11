//%%cu
#include <iostream>
#include <climits>
#include <cstdlib>
#include <chrono>
#include <omp.h>
using namespace std;

__global__ void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

void addS(int* A, int* B, int* C, int size) {
    for(int i=0;i<size;i++){
    	C[i] = A[i] + B[i];
    }

}


void initialize(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 1000;
    }
}

void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

int main() {
    int N = 10000;
    int* A, * B, * C, *D;

    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);

    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];
    D = new int[vectorSize];
    
    initialize(A, vectorSize);
    initialize(B, vectorSize);

    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);
    
    auto startSeq = chrono::steady_clock::now();
    // sequential 
    addS(A,B,D,N);
    auto endSeq = chrono::steady_clock::now();
    chrono::duration<double, micro> fp = endSeq - startSeq;
    cout << "Sequential Time: " << fp.count() << " microseconds" << endl;
    

    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);
    
    
    auto startParallel = chrono::steady_clock::now();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);
    
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);
    
    auto endParallel = chrono::steady_clock::now();
    chrono::duration<double, micro> fp1 = endParallel - startParallel;
    cout << "Parallel Time: " << fp1.count() << " microseconds" << endl;
    // cout << "Addition: ";
    // print(C, N);
    
    double SortSpeedup = fp.count() / fp1.count();
    cout << "Speedup : " << SortSpeedup << endl;
    for(int i =0;i<N;i++){
        if(C[i]!=D[i]){
            cout<<"Wrong at "<<i<<endl;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}
