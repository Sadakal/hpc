#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

// Define the number of subproblems and their size
#define NUM_SUBPROBLEMS 1000
#define SUBPROBLEM_SIZE 100

// Kernel to solve a single subproblem on the GPU
__global__ void solveSubproblem(float* input, float* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_SUBPROBLEMS) {
        // Solve the subproblem (dummy example: just sum all elements)
        float sum = 0.0f;
        for (int i = 0; i < SUBPROBLEM_SIZE; ++i) {
            sum += input[idx * SUBPROBLEM_SIZE + i];
        }
        output[idx] = sum;
    }
}

int main() {
    // Generate random input data for subproblems
    float* inputData = new float[NUM_SUBPROBLEMS * SUBPROBLEM_SIZE];
    for (int i = 0; i < NUM_SUBPROBLEMS * SUBPROBLEM_SIZE; ++i) {
        inputData[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the GPU
    float* d_inputData, * d_outputData;
    cudaMalloc((void**)&d_inputData, NUM_SUBPROBLEMS * SUBPROBLEM_SIZE * sizeof(float));
    cudaMalloc((void**)&d_outputData, NUM_SUBPROBLEMS * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_inputData, inputData, NUM_SUBPROBLEMS * SUBPROBLEM_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    int blockSize = 256;
    int numBlocks = (NUM_SUBPROBLEMS + blockSize - 1) / blockSize;

    // Measure execution time for the parallel version
    auto startPar = std::chrono::high_resolution_clock::now();

    // Launch the kernel to solve subproblems in parallel
    solveSubproblem<<<numBlocks, blockSize>>>(d_inputData, d_outputData);

    // Synchronize CUDA threads
    cudaDeviceSynchronize();

    auto endPar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationPar = endPar - startPar;

    // Copy results from device to host
    float* outputData = new float[NUM_SUBPROBLEMS];
    cudaMemcpy(outputData, d_outputData, NUM_SUBPROBLEMS * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure execution time for the sequential version
    auto startSeq = std::chrono::high_resolution_clock::now();

    // Sequential code (same as kernel code for simplicity)
    for (int i = 0; i < NUM_SUBPROBLEMS; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < SUBPROBLEM_SIZE; ++j) {
            sum += inputData[i * SUBPROBLEM_SIZE + j];
        }
        outputData[i] = sum;
    }

    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSeq = endSeq - startSeq;

    // Calculate speedup
    double speedup = durationSeq.count() / durationPar.count();

    std::cout << "Parallel Execution Time: " << durationPar.count() << " seconds" << std::endl;
    std::cout << "Sequential Execution Time: " << durationSeq.count() << " seconds" << std::endl;
    std::cout << "Speedup: " << speedup << std::endl;

    // Free allocated memory
    delete[] inputData;
    delete[] outputData;
    cudaFree(d_inputData);
    cudaFree(d_outputData);

    return 0;
}
