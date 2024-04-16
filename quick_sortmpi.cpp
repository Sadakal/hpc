#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <chrono>

using namespace std;

// Function to perform Quicksort
void quicksort(vector<int>& array, int left, int right) {
    int i = left, j = right;
    int pivot = array[(left + right) / 2];

    // Partition
    while (i <= j) {
        while (array[i] < pivot)
            i++;
        while (array[j] > pivot)
            j--;
        if (i <= j) {
            swap(array[i], array[j]);
            i++;
            j--;
        }
    }

    // Recursion
    if (left < j)
        quicksort(array, left, j);
    if (i < right)
        quicksort(array, i, right);
}

// Function to merge two sorted arrays
void merge(vector<int>& result, const vector<int>& a, const vector<int>& b) {
    size_t i = 0, j = 0, k = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j])
            result[k++] = a[i++];
        else
            result[k++] = b[j++];
    }
    while (i < a.size())
        result[k++] = a[i++];
    while (j < b.size())
        result[k++] = b[j++];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int N = 1000000; // Size of the array
    const int chunk_size = N / world_size;

    vector<int> local_array(chunk_size);
    vector<int> sorted_local_array(chunk_size);
    vector<int> merged_array(N);

    // Generate random data
    srand(time(NULL) + world_rank);
    for (int i = 0; i < chunk_size; i++) {
        local_array[i] = rand() % N;
    }

    // Measure the start time
    auto start_time = chrono::high_resolution_clock::now();

    // Perform Quicksort in parallel using OpenMP
    #pragma omp parallel
    {
        #pragma omp single nowait
        quicksort(local_array, 0, chunk_size - 1);
    }

    // Gather sorted chunks
    MPI_Gather(local_array.data(), chunk_size, MPI_INT, merged_array.data(), chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Merge sorted chunks
    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            merge(merged_array, merged_array, vector<int>(merged_array.begin() + i * chunk_size, merged_array.begin() + (i + 1) * chunk_size));
        }
    }

    // Measure the end time
    auto end_time = chrono::high_resolution_clock::now();

    // Calculate the execution time
    chrono::duration<double> elapsed = end_time - start_time;
    double parallel_execution_time = elapsed.count();

    // Calculate the serial execution time (for comparison)
    vector<int> serial_array(N);
    if (world_rank == 0) {
        srand(time(NULL)); // Reset the seed for fair comparison
        for (int i = 0; i < N; i++) {
            serial_array[i] = rand() % N;
        }
        start_time = chrono::high_resolution_clock::now();
        sort(serial_array.begin(), serial_array.end());
        end_time = chrono::high_resolution_clock::now();
        elapsed = end_time - start_time;
    }
    double serial_execution_time = elapsed.count();

    // Calculate speedup
    double speedup = serial_execution_time / parallel_execution_time;

    // Output the results
    if (world_rank == 0) {
        cout << "Serial Execution Time: " << serial_execution_time << " seconds" << endl;
        cout << "Parallel Execution Time: " << parallel_execution_time << " seconds" << endl;
        cout << "Speedup: " << speedup << "x" << endl;
    }

    MPI_Finalize();
    return 0;
}

