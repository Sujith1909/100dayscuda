#include <iostream>
#include <cmath>
#include <stdio.h>

// global is the cuda keyword that marks the function as a kernel, making it run on gpu
// Calculate global thread index:
// threadIdx.x gives thread index within block (0-255)
// blockIdx.x gives the block index
// blockDim.x is threads per block (256)
// So blockIdx.x * blockDim.x gives starting index for this block
// Adding threadIdx.x gives final global index for this thread
//
// Example: For thread 5 in block 3 with 256 threads per block:
// index = 5 + (3 * 256) = 5 + 768 = 773
//
// Each thread gets a unique index because:
// - threadIdx.x is unique within a block (0-255)
// - blockIdx.x * blockDim.x gives unique starting points for each block
// Example: If we have 1024 elements and 256 threads per block:
// Block 0: indices 0-255
// Block 1: indices 256-511  
// Block 2: indices 512-767
// Block 3: indices 768-1023
// So thread 5 in block 2 gets index: 5 + (2 * 256) = 517
// This CUDA kernel enables parallelism by:
// 1. Running multiple threads concurrently, each handling one array element
// 2. Using thread/block indices to ensure each thread works on unique data
// 3. Scaling across multiple blocks to handle large arrays
//
// For example, with N=1024 elements and 256 threads per block:
// - 4 blocks of 256 threads each will launch in parallel
// - Each thread adds one pair of numbers from arrays a and b
// - All 1024 additions happen simultaneously instead of sequentially
// - This gives a theoretical 1024x speedup vs sequential processing
__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

// Add extern "C" to make the function visible to Python
extern "C" {
    __host__ float cuda_vector_add(int *a, int *b, int *c, int N) {
        int *d_a, *d_b, *d_c;
        
        // Allocate device memory
        cudaMalloc(&d_a, N * sizeof(int));
        cudaMalloc(&d_b, N * sizeof(int));
        cudaMalloc(&d_c, N * sizeof(int));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Copy inputs to device
        cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch kernel and measure time
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        
        cudaEventRecord(start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        
        // Copy result back to host
        cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
}

int main() {
    int N = 1024;
    int size = N * sizeof(int);
    int *a, *b, *c; // host pointers
    int *d_a, *d_b, *d_c; // device pointers

    // Allocate memory on host (cpu)
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    
    // Initialize vectors
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Print input vectors
    std::cout << "Vector a: ";
    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << "\nVector b: ";
    for (int i = 0; i < N; i++) {
        std::cout << b[i] << " ";
    }
    std::cout << "\n";

    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_a, size); // similar to malloc
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); // copy data from host to device ptr
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and N threads
    vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N); // kernel launch syntax, 1 block for 256 threads, 256 threads per block
    
    // Copy the result back from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result vector
    std::cout << "Result vector c: ";
    for (int i = 0; i < N; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << "\n";

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            std::cout << "Error at index " << i << std::endl;
            return -1;
        }
    }

    std::cout << "CUDA program executed successfully!" << std::endl;

    // Clean up
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
