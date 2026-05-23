#include <cuda_runtime.h>
#include <chrono>
#include <math_functions.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float *A, float *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        B[idx] = fmaxf(A[idx], 0.0f);
    }
}

void cpu_relu_kernel(const float *A, float *B, int N)
{
    for (int i = 0; i < N; i++)
    {
        B[i] = fmaxf(A[i], 0.0f);
    }
    return;
}

int main()
{
    // Initialize random vector for relu operation:
    int n = 1 << 20;  // Size of vectors (1 million elements)
    size_t size = n * sizeof(float);
    std::vector<float> h_A(n, 1.0f), h_B(n, 0.0f);

    // Device memory pointers
    float *d_A, *d_B;

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    auto start = std::chrono::high_resolution_clock::now();
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuDuration = end - start;
    std::cout << "GPU time: " << gpuDuration.count() << " seconds" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);

    // Cpu test
    start = std::chrono::high_resolution_clock::now();

    cpu_relu_kernel(h_A.data(), h_B.data(), n);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = end - start;
    std::cout << "CPU time: " << cpuDuration.count() << " seconds" << std::endl;

    return 0;
}
