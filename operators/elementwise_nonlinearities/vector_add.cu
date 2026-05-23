#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

// CUDA kernel: Runs on the GPU

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    // Global thread index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Boundary check
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void cpuVectorAdd(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

void runTest(int threadsPerBlock)
{
    int n = 1 << 20;  // Size of vectors (1 million elements)
    size_t size = n * sizeof(float);

    std::cout << "Vector size: " << n << " elements" << std::endl;
    // Allocate host memory
    std::vector<float> h_A(n, 1.0f), h_B(n, 2.0f), h_C(n, 0.0f), h_ref(n, 0.0f);

    // Device memory pointers
    float *d_A, *d_B, *d_C;
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Define block and grid sizes - in a loop to test different configurations

    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    auto start = std::chrono::high_resolution_clock::now();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();  // Wait for the GPU to finish
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuDuration = end - start;
    std::cout << "GPU time: " << gpuDuration.count() << " seconds" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // CPU computation for reference
    start = std::chrono::high_resolution_clock::now();
    cpuVectorAdd(h_A.data(), h_B.data(), h_ref.data(), n);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = end - start;
    std::cout << "CPU time: " << cpuDuration.count() << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int n = 1 << 20;  // Size of vectors (1 million elements)
    size_t size = n * sizeof(float);

    std::cout << "Vector size: " << n << " elements" << std::endl;
    // Allocate host memory
    std::vector<float> h_A(n, 1.0f), h_B(n, 2.0f), h_C(n, 0.0f), h_ref(n, 0.0f);

    // Device memory pointers
    float *d_A, *d_B, *d_C;
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Define block and grid sizes - in a loop to test different configurations

    //
    std::vector<int> threadsPerBlock = {64, 128, 256, 512, 1024};
    for (int tpb : threadsPerBlock)
    {
        std::cout << "Testing with " << tpb << " threads per block..." << std::endl;
        runTest(tpb);
    }
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // // Launch the kernel
    // auto start = std::chrono::high_resolution_clock::now();
    // vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    // cudaDeviceSynchronize(); // Wait for the GPU to finish
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> gpuDuration = end - start;
    // std::cout << "GPU time: " << gpuDuration.count() << " seconds" << std::endl;

    // // Copy result back to host
    // cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // // CPU computation for reference
    // start = std::chrono::high_resolution_clock::now();
    // cpuVectorAdd(h_A.data(), h_B.data(), h_ref.data(), n);
    // end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> cpuDuration = end - start;
    // std::cout << "CPU time: " << cpuDuration.count() << " seconds" << std::endl;

    // // Free device memory
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
    // return 0;
}
