# Lab 1: Saxpy in "CUDA Python"

* Implement saxpy in "CUDA Python"
* The lab is broken down into four small exercises
* We will provide guidelines and hints along the way
* lab1/saxpy.py

# Exercise 1


## Host -> Device

- `d_ary = cuda.to_device(ary)`
    - `cudaMalloc(size);`
    - `cudaMemcpy(devary, hstary, size, cudaMemcpyHostToDevice);`

## Host -> Device (allocate only, no copy)

- `d_ary = cuda.to_device(ary, copy=False)`
    - `cudaMalloc(size);`


# Exercise 2

## Kernel Launch

- `griddim`: tuple of 1-2 ints
- `blockdim`: tuple of 1-3 ints
    - `dim3 griddim, blockdim;
- `a_kernel[griddim, blockddim](arg0, arg1)`
    - `a_kernel<<<griddim, blockdim>>>(arg0, arg1);`


# Exercise 3

##  Device -> Host

- `d_ary.to_host()`
    - `cudaMemcpy(hstary, devary, size, cudaMemcpyHostToDevice);`


# Exercise 4


## Inside the Kernel

- `cuda.threadIdx`, `cuda.blockIdx`, `cuda.blockDim`
    - `threadIdx`, `blockIdx`, `blockDim`
    
```python
i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
```

