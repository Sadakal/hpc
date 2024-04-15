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

    // Measure execution time using chrono
    if (world_rank == 0) {
        auto start_time = chrono::high_resolution_clock::now();

        // Do something with the sorted merged array

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;
        cout << "Time taken: " << elapsed.count() << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
