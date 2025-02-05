import numpy as np
import time
from numba import cuda
from ctypes import *
import os

# Load the CUDA library
cuda_lib = CDLL('./vector_add.so')  # You'll need to compile the CUDA code first

def cpu_vector_add(a, b):
    start_time = time.time()
    c = a + b
    end_time = time.time()
    return c, (end_time - start_time) * 1000  # Convert to milliseconds

def main():
    # Initialize data
    N = 1000000
    a = np.random.randint(0, 100, size=N, dtype=np.int32)
    b = np.random.randint(0, 100, size=N, dtype=np.int32)
    c = np.zeros(N, dtype=np.int32)
    
    # CPU Version
    cpu_result, cpu_time = cpu_vector_add(a, b)
    print(f"CPU Time: {cpu_time:.2f} ms")
    
    # CUDA Version
    cuda_lib.cuda_vector_add.restype = c_float
    cuda_time = cuda_lib.cuda_vector_add(
        a.ctypes.data_as(POINTER(c_int)),
        b.ctypes.data_as(POINTER(c_int)),
        c.ctypes.data_as(POINTER(c_int)),
        c_int(N)
    )
    print(f"CUDA Time: {cuda_time:.2f} ms")
    
    # Verify results
    np.testing.assert_array_equal(c, cpu_result)
    print("Results match!")

if __name__ == "__main__":
    main()