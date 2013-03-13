import numpy as np
import math
import contextlib
import numbapro
from numbapro import cuda
from numbapro.cudalib import curand

# PRNG constants
A = 1664525
C = 1013904223

@cuda.jit("void(double[:,:], double, double, double, double[:], uint32[:])")
def cu_monte_carlo_pricer(paths, dt, c0, c1, normdist, seed):
    # short for cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    i = cuda.grid(1)
    if i >= paths.shape[0]:
        return
    randnum = seed[i]
    for j in range(1, paths.shape[1]):         # foreach time step
        elt = randnum % normdist.shape[0]
        if elt < 0 :
            elt = -elt
        noise = normdist[elt]
        paths[i, j] = paths[i, j - 1] * np.exp(c0 * dt + c1 * noise)
        # generate next random number
        randnum = randnum * A + C

def monte_carlo_pricer(paths, dt, interest, volatility):
    n = paths.shape[0]
    blksz = 512
    gridsz = int(math.ceil(float(n) / blksz))

    stream = cuda.stream()
    prng = curand.PRNG(curand.PRNG.MRG32K3A, stream=stream)
    qrng = curand.QRNG(curand.QRNG.SOBOL32, stream=stream)

    d_normdist = cuda.device_array(n, dtype=np.double, stream=stream)
    d_seed = cuda.device_array(n, dtype=np.uint32, stream=stream)

    prng.normal(d_normdist, 0, 1)
    qrng.generate(d_seed)

    d_paths = cuda.to_device(paths, stream=stream)
    
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * math.sqrt(dt)

    griddim = gridsz, 1
    blockdim = blksz, 1, 1
    cu_monte_carlo_pricer[griddim, blockdim, stream](d_paths, dt, c0, c1,
                                                     d_normdist, d_seed)

    d_paths.to_host(stream)

    stream.synchronize()

if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer)
