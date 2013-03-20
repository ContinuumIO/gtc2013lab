% Using Python to Speed-up Applications with GPUs
% Continuum Analytics
% Travis Oliphant & Siu Kwan Lam

# Agenda

* Introduction
    - Python
    - Numba
    - NumbaPro
* Lab 1
* Lab 2

# Introduction

## Why Python?

Pros

* A dynamic scripting language
* Rapid development
* Rich libraries --- NumPy, SciPy
* Great glue language to connect native libraries

Cons

* Hard to parallellize because of the GIL
* Slow execution speed

# Our Solution: Speedup Python with Numba

## Numba

* An opensource **JIT** compiler for **array-oriented programming** in CPython
* Turn numerical loops into fast native code
* Maximize hardware utilization
* Just add a decorator to your compute intensive function

# Mandelbrot in Numba

```python
from numba import autojit
@autojit  # <--- All we need to add
def mandel(x, y, max_iters):
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return 255
```

* Over **170x** speedup


# Need More Speed? Use NumbaPro "CUDA Python"

* Our commerical product enables parallel compute on GPU.

```python
from numbapro import cuda, uint8, f8, uint32
# use CUDA jit
@cuda.jit(argtypes=[uint8[:,:], f8, f8, f8, f8, uint32])
def mandel_kernel(image, min_x, max_x, 
                  min_y, max_y, iters):
	height = image.shape[0]
	width = image.shape[1]
	pixel_size_x = (max_x - min_x) / width
	pixel_size_y = (max_y - min_y) / height
    # access thread indices
	x = cuda.threadIdx.x + \    
                cuda.blockIdx.x * cuda.blockDim.x
	y = cuda.threadIdx.y + \        
                cuda.blockIdx.y * cuda.blockDim.y
    # truncated ...
```

* **1255x** faster on K20
