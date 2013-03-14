# Lab 1
## Saxpy in "CUDA Python"

# Quick Lesson to CUDA Python

* Similar to CUDA-C

# Memory Transfer

* Host->Device
    - `d_ary = cuda.to_device(ary)`
* Host->Device (allocate only, no copy)
    - `d_ary = cuda.to_device(ary, copy=False)`
* Device->Host
    - `d_ary.to_host()`
    
# Compile and Launch

* Decorate kernel
    - `cuda.autojit`, `cuda.jit`
* Kernel launch
    - `a_kernel[griddim, blockddim](arg0, arg1)`
    - similar to C: `a_kernel<<<griddim, blockdim>>>(arg0, arg1)`
    - `griddim`: tuple of 1-2 ints
    - `blockdim`: tuple of 1-3 ints
* `threadIdx`, `blockIdx`, `blockDim` -> `cuda.threadIdx`, `cuda.blockIdx`, `cuda.blockDim`
    - `i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x`



