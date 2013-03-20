# CUDA C Memory transfer

* cudaMalloc(devptr, size)
    * devptr: output device pointer
    * size: allocation size
* cudaFree(devptr)
* cudaMemcpy(dst, src, sz, flag)
    * dst: destination pointer
    * src: souce pointer
    * sz: transfer byte size
    * flag: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost


# CUDA C Kernel Launch

```C
dim3 griddim;
dim3 blockdim;
a_kernel<<<griddim, blockdim>>>(arg0, arg1, ..., argN);
```

# CUDA C Putting Everything Together
```C
float host_ary[n];
// host -> device
float device_ary;
cudaMalloc(&device_ary, n * sizeof(float));
cudaMemcpy(device_ary, host_ary, cudaMemcpyHostToDevice);
// launch kernel
dim3 blockdim(256);
dim3 griddim(n / 256); // assume n is multiple of 256
inplace_square<<griddim, blockdim>>>(device_ary);
// device -> host
cudaMemcpy(host_ary, device_ary, cudaMemcpyDeviceToHost);
cudaFree(device_ary);
```
