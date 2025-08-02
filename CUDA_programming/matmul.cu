// simple matmul on GPU
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    }

#define CHECK_LAST_ERROR() CHECK_CUDA(cudaGetLastError())

#define BLOCK_X 16
#define BLOCK_Y 8

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


// a - M X N b - N X K c - M X K
void matmul_cpu(float *a, float *b, float *c, int M, int N, int K)
{
    for(int i=0;i < M; i++)
    {
        for(int j=0;j < K; j++)
        {
            int out_idx = j + K * i;
            float sum = 0.0f;
            for(int k = 0; k < N; k++)
            {
                int in_idx1 = k + N * i;
                int in_idx2 = j + K * k;
                sum += a[in_idx1] * b[in_idx2];
            }
            c[out_idx] = sum;
        }
    }
}

__global__ void matmul_gpu(float *a, float *b, float *c, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K)
    {
        float sum = 0.0f;
        for(int l=0;l < N; l++)
            sum += a[l + N * row] * b[col + K * l];
        c[row * K + col] = sum;
    }


}

void print_mat(float *a, int M, int N)
{
    printf("\n");
    for(int i=0;i<M;i++)
    {
        for(int j=0;j < N; j++)
        {
            printf("%f ", a[j+i*N] );
        }
        printf("\n");
    }
}

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int M = 1024;
    int N = 864;
    int K = 384;
    h_a = (float*)malloc(M * N * sizeof(float));
    h_b = (float*)malloc(N * K * sizeof(float));
    h_c_cpu = (float*)malloc(M * K * sizeof(float));
    h_c_gpu = (float*)malloc(M*K * sizeof(float));
    cudaMalloc(&d_a, M*N*sizeof(float));
    cudaMalloc(&d_b, N*K*sizeof(float));
    cudaMalloc(&d_c, M*K*sizeof(float));
    init_vector(h_a, M*N);
    init_vector(h_b,N*K);

    int num_blocks_y = (M + BLOCK_Y - 1) / BLOCK_Y;
    int num_blocks_x = (K + BLOCK_X - 1) / BLOCK_X;

    dim3 num_blocks(num_blocks_x, num_blocks_y);
    dim3 block_size(BLOCK_X, BLOCK_Y);

    cudaMemcpy(d_a, h_a, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float),cudaMemcpyHostToDevice);

    //print_mat(h_a, M, N);
    //print_mat(h_b, N, K);
    printf("warming up for 3 iterations!\n");
    for(int i=0;i<3;i++)
    {
        // CPU implementation
        matmul_cpu(h_a, h_b, h_c_cpu, M, N, K);
        //printf("printing cpu matrix result!");
        //print_mat(h_c_cpu, M, K);
        // GPU implementation
        matmul_gpu<<<num_blocks, block_size>>>(d_a, d_b, d_c, M, N, K);
        CHECK_LAST_ERROR();
        cudaDeviceSynchronize();
    }

    printf("benchmarking cpu matmul implementation!\n");
    double cpu_total_time = 0.0;
    for(int i=0;i<10;i++)
    {
        double start_time = get_time();
        matmul_cpu(h_a, h_b, h_c_cpu, M, N, K);
        double end_time = get_time();
        cpu_total_time += (end_time - start_time);
    }
    double cpu_avg_time = cpu_total_time / 10.0;

    printf("Benchmarking GPU implementation!\n");
    double gpu_total_time = 0.0;
    for(int i=0;i<10;i++)
    {
        double start_time = get_time();
        matmul_gpu<<<num_blocks, block_size>>>(d_a, d_b, d_c, M, N, K);
        CHECK_LAST_ERROR();
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += (end_time - start_time);
    }
    double gpu_avg_time = gpu_total_time / 10.0;

    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    printf("verification of results - matmul !\n");
    cudaMemcpy(h_c_gpu, d_c, M*K*sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for(int i=0;i<N;i++)
    {
        if (fabsf(h_c_cpu[i] - h_c_gpu[i]) > 1e-3)
        {
            printf("mismatch at %d, cpu val = %.2f, gpu_val = %.2f\n", i, h_c_cpu[i],h_c_gpu[i]);
            correct = false;
            break;
        }
    }
    //printf("printing gpu matrix result");
    //print_mat(h_c_gpu, M, K);

    printf("Results are %s\n", correct ? "correct":"incorrect");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}