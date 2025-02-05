# 100dayscuda

day 1: 
Today's learnings from vector addition:

1. CUDA Kernel Structure:
   - `__global__` keyword marks a function to run on GPU
   - Each thread gets unique index via: threadIdx.x + blockIdx.x * blockDim.x
   - Example: Thread 5 in block 3 with 256 threads gets index: 5 + (3 * 256) = 773

2. Thread/Block Organization:
   - Threads are grouped into blocks (e.g. 256 threads per block)
   - Multiple blocks form a grid
   - For 1024 elements with 256 threads/block:
     - Block 0: indices 0-255
     - Block 1: indices 256-511
     - Block 2: indices 512-767
     - Block 3: indices 768-1023

3. Memory Management:
   - Need to explicitly allocate GPU memory with cudaMalloc()
   - Copy data between CPU (host) and GPU (device) using cudaMemcpy()
   - Remember to free GPU memory with cudaFree()

4. Performance Results:
   - CPU Time: 2.17 ms
   - CUDA Time: 0.14 ms
   - ~15x speedup using parallel GPU computation

5. Key Benefits:
   - Massive parallelism: Each thread handles one array element
   - All additions happen simultaneously instead of sequentially
   - Scales well with larger arrays by using multiple blocks
