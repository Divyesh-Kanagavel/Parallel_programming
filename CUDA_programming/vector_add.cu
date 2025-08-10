// vector addition in cuda

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000000 // vector size : 1 million
# define BLOCK_SIZE 256 // number of blocks per grid

// cpu vector addition
void vector_add_cpu(float* a, float* b, float* c, int n)
{
    for(int i=0;i<n;i++)
    {
        c[i] = a[i] + b[i]; // sequential addition
    }
}

__global__ void vector_add_gpu(float* a, float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

// init vector with random values
void init_vector(float* a, int n)
{
    for(int i=0;i<n;i++)
    {
        a[i] = (float)rand() / RAND_MAX;
    }
}

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu; // host tensors
    float *d_a, *d_b, *d_c; //device tensors

    size_t SIZE = N * sizeof(float);
    // allocate memory for the tensors in host machine
    h_a = (float*)malloc(SIZE);
    h_b = (float*)malloc(SIZE);
    h_c_cpu = (float*)malloc(SIZE);
    h_c_gpu = (float*)malloc(SIZE);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // allocate memory for tensors in the device
    cudaMalloc(&d_a, SIZE);
    cudaMalloc(&d_b , SIZE);
    cudaMalloc(&d_c, SIZE);

    // copy tensors from host to device
    cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;

    printf("performing warm-up runs!\n"); // important to get accurate runtime, because in the first few runs, there are memory caching, page fault fixes etc done.
    for(int i=0;i<5;i++)
    {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU implementation!\n");
    double cpu_total_time = 0.0;
    for(int i=0;i<20;i++)
    {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += (end_time - start_time);
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation!\n");
    double gpu_total_time = 0.0;
    for(int i=0;i<20;i++)
    {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += (end_time - start_time);
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    printf("verification of results!\n");
    cudaMemcpy(h_c_gpu, d_c, SIZE, cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int i=0;i<N;i++)
    {
        if (fabsf(h_c_cpu[i] - h_c_gpu[i]) > 1e-5)
        {
            printf("Mismatch at i=%d: CPU = %f, GPU = %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct":"incorrect");

    // free memories
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

}