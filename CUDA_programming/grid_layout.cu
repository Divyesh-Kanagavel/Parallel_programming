#include <stdio.h>
#include <cuda_runtime.h>

__global__ void whoami(void)
{
    int blockid = blockIdx.x +
                  blockIdx.y * gridDim.x +
                  blockIdx.z * gridDim.y * gridDim.x;
    int block_offset = blockid * blockDim.x * blockDim.y * blockDim.z;

    int thread_offset = threadIdx.x +
                   threadIdx.y * blockDim.x +
                   threadIdx.z * blockDim.y * blockDim.x;
    int id = thread_offset + block_offset;

    printf("id, thread_offset, blockid, blockoffset = %d,%d,%d,%d\n", id, thread_offset, blockid, block_offset);
    printf("blockidx.x, blockidx.y, blockidx.z = %d,%d,%d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("threadidx.x, threadidx.y, threadidx.z = %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z);


}

int main(int argc, char** argv)
{
    int b_x = 2, b_y = 3, b_z = 4;
    int t_x = 4, t_y = 4, t_z = 4;

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>> ();
    cudaDeviceSynchronize();
}