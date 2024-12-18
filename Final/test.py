# import torch
# a = torch.rand(10000, 10000)
# b = torch.rand(10000, 10000)
# c = torch.matmul(a, b)  # GPU/CPU computation based on tensor device

import numpy as np
from time import perf_counter

size = 10000  # Large matrix
A = np.random.rand(size, size)
B = np.random.rand(size, size)

start_time = perf_counter()
C = np.dot(A, B)  # Matrix multiplication
end_time = perf_counter()

print(f"Time taken: {end_time - start_time:.4f} seconds")
