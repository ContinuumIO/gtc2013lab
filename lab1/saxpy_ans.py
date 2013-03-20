from numbapro import cuda
from numba import *
import numpy as np
import math
from timeit import default_timer as time

@cuda.autojit
def saxpy(Out, a, X, Y):
    "Compute Out = a * X + Y"
    # ------ Exercise --------
    # Complete this kernel
    # threadIdx ---> cuda.threadIdx
    # blockIdx  ---> cuda.blockIdx
    # blockDim  ---> cuda.blockDim
    i = cuda.threadIdx.x  + cuda.blockIdx.x * cuda.blockDim.x
    Out[i] = a * X[i] + Y[i]

def main():
    # Prepare data
    thread_per_block = 512
    block_per_grid = 10
    n = thread_per_block * block_per_grid
    a = 1.2345
    X = np.random.random(n).astype(np.float32)
    Y = np.random.random(n).astype(np.float32)
    Out = np.empty_like(X)

    # ------ Exercise --------
    # Host->Device
    # Complete the transfer for X, Y, and Out
    dX = cuda.to_device(X)
    dY = cuda.to_device(Y)
    dOut = cuda.to_device(Out, copy=False)

    # Kernel launch
    blockdim = thread_per_block, 1, 1
    griddim = block_per_grid, 1

    # ------ Exercise --------
    # Kernel launch
    # Complete the kernel launch for saxpy
    saxpy[griddim, blockdim](dOut, a, dX, dY)

    # ------ Exercise --------
    # Device->Host
    # Complete the transfer for dOut
    dOut.to_host()

    print('-- Result --')
    print(Out)
    # Verify
    print("verify: %s" % np.allclose(a * X + Y, Out))

if __name__ == '__main__':
    main()
