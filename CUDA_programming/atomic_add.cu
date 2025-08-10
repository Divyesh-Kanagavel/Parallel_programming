#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

__global__ void incrementNonAtomic(int *counter)
{
  // not locked and unlocked -> data race condition
  int old = *counter;
  int new_value = old + 1;
  *counter = new_value;
}



__global__ void incrementAtomic(int *counter)
{
  // atomic add -> prevents data race condition
  int a = atomicAdd(counter, 1);
}

int main()
{
  int h_counterAtomic = 0;
  int h_counterNonAtomic = 0;
  int *d_counterAtomic;
  int *d_counterNonAtomic;

  cudaMalloc((void**)&d_counterAtomic, sizeof(int));
  cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));

  cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);

  // launch kernels
  incrementAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);
  incrementNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);

  // copy the data back to host
  cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Non Atomic counter value = %d\n", h_counterNonAtomic);
  printf("Atomic counter value = %d\n", h_counterAtomic);

  // free device memory
  cudaFree(d_counterAtomic);
  cudaFree(d_counterNonAtomic);

  return 0;

}