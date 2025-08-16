#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// kernel1
__global__ void kernel1(float *data, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    data[i] *= 2.0f;
  }
}

// kernel 2
__global__ void kernel2(float *data, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    data[i] += 1.0f;
  }
}

void CUDART_CB myStreamCallBack(cudaStream_t stream, cudaError_t status, void *userData)
{
  printf("Stream callback : Operation completed!\n");
}



int main()
{
  const int N = 1000000;
  size_t size = N * sizeof(float);
  float *h_a, *d_a;

  cudaStream_t stream1, stream2;
  cudaEvent_t event;

  CHECK_CUDA_ERROR(cudaMallocHost(&h_a, size)); //pinned memory for faster transfers
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size)); // memory allocation in device



  for(int i=0;i<N;i++)
  {
    h_a[i] = static_cast<float>(i);
  }

  int leastPriority, greatestPriority;
  CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
  CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));


  CHECK_CUDA_ERROR(cudaEventCreate(&event));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1));

  kernel1<<<(N + 255)/256, 256, 0, stream1>>>(d_a, N);
  
  CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));

  CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

  kernel2<<<(N + 255)/256, 256, 0, stream2>>>(d_a, N);

  // add callback to stream2
  CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, myStreamCallBack, NULL, 0));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_a, d_a, size, cudaMemcpyDeviceToHost, stream2));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

  // verification of results

  for(int i=0;i<N;i++)
  {
    float expected = 2.0f * static_cast<float>(i) + 1.0f;
    if (fabs(h_a[i]-expected) > 1e-5)
    {
      fprintf(stderr, "Result verifiation failed!\n");
      exit(EXIT_FAILURE);

    }
  }
  printf("Results verified!\n");

  return 0;
}