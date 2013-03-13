from numbapro import cuda
from numba import *
import numpy as np
import math
from timeit import default_timer as time

@cuda.autojit
def saxpy(Out, X, Y, Z):
    "Compute Out = X * Y + Z"
    # ------ Exercise --------
    # Complete this kernel
    # threadIdx ---> cuda.threadIdx
    # blockIdx  ---> cuda.blockIdx
    # blockDim  ---> cuda.blockDim


def main():
    # Prepare data
    thread_per_block = 512
    block_per_grid = 10
    n = thread_per_block * block_per_grid
    X = np.random.random(n)
    Y = np.random.random(n)
    Z = np.random.random(n)
    Out = np.empty_like(X)

    # ------ Exercise --------
    # Host->Device
    # Complete the transfer for Y, Z, and Out

    # Kernel launch
    blockdim = thread_per_block, 1, 1
    griddim = block_per_grid, 1

    
    # ------ Exercise --------
    # Kernel launch
    # Complete the kernel launch for saxpy


    # ------ Exercise --------
    # Device->Host
    # Complete the transfer for dOut


    print('-- Result --')
    print(Out)
    # Verify
    print("verify: %s" % np.allclose(X * Y + Z, Out))

if __name__ == '__main__':
    main()
