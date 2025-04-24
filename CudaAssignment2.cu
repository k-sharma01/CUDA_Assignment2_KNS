/*
* Kirin Sharma
* CS-425 Advanced Architecture
* CUDA Assignment 2
*
* This program creates and runs a CUDA kernel to multiply a matrix and a vector and store to a result vector. Based on how 
* the matrix and vector are initialized, the resulting vector should contain 10 of the same value:
*  the sum of squares 1-10 which should come out to 385
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>

using namespace std;

// CUDA kernel function to multiply the matrix and the vector and store to a result vector
__global__ void multiply(int *matrix, int *vector, int* result, int size)
{
    // Used to identify which thread should work on which data and initialize shared memory
    int row = blockIdx.x;
    int col = threadIdx.x;
    __shared__ int partialSum[16];

    // Have the current thread compute its partial sum and store it to shared memory, then sync threads in the block (only use threads 1-10, rest are 0)
    if(col < size) {
        partialSum[col] = matrix[row * size + col] * vector[col];
    } else {
        partialSum[col] = 0;
    }
    __syncthreads();

    // Compute total sum in parallel from the shared memory
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if(col < stride){
            partialSum[col] += partialSum[col + stride];
        }
        __syncthreads();
    }

    // Store the final result to the result matrix
    if(col == 0) {
        result[row] = partialSum[0];
    }
}

// Helper function to print a matrix
void printMatrix(int* matrix, int size) {
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            cout << matrix[size * i + j] << "   ";
        }
        cout << "\n";
    }
}

// Helper function to print a vector
void printVector(int* vector, int size) {
    for(int i = 0; i < size; i++) {
        cout << vector[i] << "   ";
    }
}

int main()
{
    // Allocate unified memory for the (flattened) 2-D matrix and vectors
	int size = 10;
	int *matrix;
	int *vector;
    int *result;
    cudaMallocManaged(&matrix, size * size * sizeof(int));
    cudaMallocManaged(&vector, size * sizeof(int));
    cudaMallocManaged(&result, size * sizeof(int));

    // Initialize the matrix and vector
    for(int i = 0; i < size; i++) {
        vector[i] = i + 1;
        for(int j = 0; j < size; j++) {
            matrix[i * size + j] = j + 1;
        }
    }

    // Print the starting vector
    cout << "Starting Vector:\n";
    printVector(vector, size);
    cout << "\n\n";

    // Print the starting matrix
    cout << "Starting Matrix:\n";
    printMatrix(matrix, size);
    cout << "\n\n";

    // Specify 10 blocks with 16 threads each (power of 2 even though only 10 really used per block)
    int numBlocks = 10;
    int threadsPerBlock = 16;

    // Launch the cuda kernel
    multiply<<<numBlocks, threadsPerBlock>>>(matrix, vector, result, size);
    cudaDeviceSynchronize();

    // Print the result vector
    cout << "Result Vector:\n";
    printVector(result, size);

    // Free cuda memory
    cudaFree(matrix);
    cudaFree(vector);
    cudaFree(result);

    return 0;
    
} // end main
