#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* funcName, const char* fileName, int LineNum)
{
  if (err!=cudaSuccess)
  {
    fprintf(stderr, "CUDA error at %s:%d code = %d(%s) \"%s\" \n", fileName, LineNum, static_cast<int>(err), cudaGetErrorString(err), funcName);
    exit(EXIT_FAILURE);
  }
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int NumElements)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NumElements)
  {
    C[i] = A[i] + B[i];
  }
}

int main()
{
  int NumElements = 50000;
  size_t size = NumElements * sizeof(float);
  float *h_a, *h_b, *h_c; // buffers in host
  float *d_a, *d_b, *d_c; // buffers in device
  cudaStream_t stream1, stream2;

  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c = (float *)malloc(size);

  for(int i=0;i<NumElements;i++)
  {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

  cudaEvent_t event;
  CHECK_CUDA_ERROR(cudaEventCreate(&event));
  

  // memcpy from host to device in two cuda_streams - the control is passed back to cpu while the mem copy happens utilizing the gpus in the background

  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice,stream1));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice,stream2));

  CHECK_CUDA_ERROR(cudaEventRecord(event,stream2));

  CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream1, event,0));

  int threadsPerBlock = 256;
  int numBlocks = (NumElements + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, NumElements);


  CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c, d_c, size,cudaMemcpyDeviceToHost, stream1));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

  free(h_a);
  free(h_b);
  free(h_c);
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
  CHECK_CUDA_ERROR(cudaEventDestroy(event));

  
  return 0;
}