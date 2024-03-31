#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

// Function to perform Bubble Sort
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Function to perform Parallel Bubble Sort
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
#pragma omp parallel
    {
        for (int i = 0; i < n - 1; i++) {
#pragma omp for
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }
        }
    }
}


// Function to perform Merge Sort
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}


// Function to perform Parallel Merge Sort
void parallelMerge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

#pragma omp parallel for
    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];

#pragma omp parallel for
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            parallelMergeSort(arr, l, m);
#pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        parallelMerge(arr, l, m, r);
    }
}

vector<int> generateRandomArray(int n) {
    vector<int> arr;
    arr.reserve(n);
    srand(static_cast<unsigned>(time(nullptr))); 

    for (int i = 0; i < n; ++i) {
        arr.push_back(rand() % n + 1); 
    }
    return arr;
}

int main() {
    const int n = 10000;

    // Generate a random array
    vector<int> arr = generateRandomArray(n);
    vector<int> arr1(arr), arr2(arr), arr3(arr), arr4(arr);

    // Measure time for Bubble Sort
    auto start_bubble_seq = chrono::high_resolution_clock::now();
    bubbleSort(arr1);
    auto end_bubble_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_bubble_seq = end_bubble_seq - start_bubble_seq;
    cout << "Bubble Sort Time (Sequential): " << duration_bubble_seq.count() << " seconds\n";

    // Measure time for Parallel Bubble Sort
    auto start_bubble_par = chrono::high_resolution_clock::now();
    parallelBubbleSort(arr2);
    auto end_bubble_par = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_bubble_par = end_bubble_par - start_bubble_par;
    cout << "Parallel Bubble Sort Time: " << duration_bubble_par.count() << " seconds\n";

    // Calculate speedup for Bubble Sort
    double speedup_bubble = duration_bubble_seq.count() / duration_bubble_par.count();
    cout << "Speedup (Bubble Sort): " << speedup_bubble << endl;

    // Generate a new random array for the following sorts
    arr = generateRandomArray(n);

    // Measure time for Merge Sort
    auto start_merge_seq = chrono::high_resolution_clock::now();
    mergeSort(arr3, 0, n - 1);
    auto end_merge_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_merge_seq = end_merge_seq - start_merge_seq;
    cout << "Merge Sort Time (Sequential): " << duration_merge_seq.count() << " seconds\n";

    // Measure time for Parallel Merge Sort
    auto start_merge_par = chrono::high_resolution_clock::now();
    parallelMergeSort(arr4, 0, n - 1);
    auto end_merge_par = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_merge_par = end_merge_par - start_merge_par;
    cout << "Parallel Merge Sort Time: " << duration_merge_par.count() << " seconds\n";

    // Calculate speedup for Merge Sort
    double speedup_merge = duration_merge_seq.count() / duration_merge_par.count();
    cout << "Speedup (Merge Sort): " << speedup_merge << endl;

    return 0;
}
