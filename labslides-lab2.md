# Lab 2

## A "Monty" Carlo Option Pricer in NumbaPro CU API

# Quick Lesson on NumbaPro CU API

* CU = Compute Unit
* A OpenCL-like API to heterogeneous parallel computing
* Instantiate a CU for 'gpu' or 'cpu'
    - `cu = CU('gpu')`
* Transfer data to the CU
    - `d_ary = cu.input(ary)`
    - `d_ary = cu.output(ary)`
    - `d_ary = cu.inout(ary)`
    - `d_ary = cu.scratch(shape=arraylen, dtype=np.float32)`
    - `d_ary = cu.scratch_like(ary)`

* Enqueue kernels to the CU
    - `cu.enqueue(kernel, ntid=number_of_threads, args=(arg0, arg1))`
* The kernel runs asynchronously
* Wait for the kernel to complete
    - `cu.wait()`

# A Numpy Implementation

```python
import numpy as np
from math import sqrt, exp
from timeit import default_timer as timer

def step(dt, prices, c0, c1, noises):
    return prices * np.exp(c0 * dt + c1 * noises)

def monte_carlo_pricer(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in xrange(1, paths.shape[1]):
        prices = paths[:, j - 1]
        noises = np.random.normal(0., 1., prices.size)
        paths[:, j] = step(dt, prices, c0, c1, noises)

```
