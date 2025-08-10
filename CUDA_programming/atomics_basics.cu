#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    }

#define CHECK_LAST_ERROR() CHECK_CUDA(cudaGetLastError())
// mutex structure
struct Mutex
{
  int *lock;
};

// initialize the mutex
__host__ void initMutex(Mutex *m)
{
  cudaMalloc((void**)&m->lock, sizeof(int));
  int initial = 0; // initally unlocked
  CHECK_CUDA(cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice));
}

__device__ void lock(Mutex *m)
{
  while(atomicCAS(m->lock, 0, 1)!=0)
  {
    // wait 
  }
}

__device__ void unlock(Mutex *m)
{
  atomicExch(m->lock, 0); // release the mutex
}

// CUDA kernel
__global__ void mutexKernel(int *counter, Mutex *m)
{
  lock(m);
  // critical section
  int old = *counter;
  *counter = old + 1;
  unlock(m);
}

// main function
int main()
{
  Mutex m;
  initMutex(&m);
  int *d_counter;
  cudaMalloc((void**)&d_counter, sizeof(int));
  int initial = 0;
  cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);

  // launch kernel with multiple threads
  mutexKernel<<<1,1000>>>(d_counter, &m);

  int result;
  cudaMemcpy(&result, d_counter,sizeof(int), cudaMemcpyDeviceToHost);
  printf("Result: %d\n", result);

  cudaFree(m.lock);
  cudaFree(d_counter);

  return 0;

}