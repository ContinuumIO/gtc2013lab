# "CUDA Python" Too Low-Level?

- "CUDA Python" API may be too close to CUDA-C
- We want simpler API
- An API that is closer to array-expression

# NumbaPro Compute Unit (CU) API

- Heterogeneous parallel programming for GPU, CPU, ...
- Execute kernels asynchronously

# A Saxpy Example

```python
def product(tid, A, B, Prod):
    Prod[tid] = A[tid] * B[tid]

def sum(tid, A, B, Sum):
    Sum[tid] = A[tid] + B[tid]

cu = CU('gpu') # or CU('cpu')
... # prepare data
cu.enqueue(product, ntid=dProd.size,
           args=(dA, dB, dProd))
cu.enqueue(sum, 	ntid=dSum.size,
           args=(dProd, dC, dSum))
... # do something while waiting?
cu.wait()

cu.close() # destroy the compute unit
```
