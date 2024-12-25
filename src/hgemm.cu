#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <stdio.h>

// Enable Tensor Core operations
#pragma enable_tf32_tensor_core_optimization

#define N 1024
#define BLOCK_SIZE 16
#define WARP_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

// WMMA matrix tiles
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Add non-Tensor Core version of HGEMM
__global__ void hgemm_normal(const half* A, const half* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            float a_val = __half2float(A[row * N + k]);
            float b_val = __half2float(B[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

__global__ void hgemm_tensor_core(const half* A, const half* B, float* C) {
    using namespace nvcuda;
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Calculate block position
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize accumulator with zeros
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int k = 0; k < N; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Load the inputs
        wmma::load_matrix_sync(a_frag, A + aRow * N + aCol, N);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
}

int main() {
    half *a_h, *b_h;
    float *c_h;
    half *a_d, *b_d;
    float *c_d;

    // Allocate host memory
    a_h = new half[N * N];
    b_h = new half[N * N];
    c_h = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a_h[i] = __float2half(1.0f);
        b_h[i] = __float2half(1.0f);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&a_d, N * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&b_d, N * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&c_d, N * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(half), cudaMemcpyHostToDevice));

    // Allocate second output array for normal version
    float *c_normal_d;
    CUDA_CHECK(cudaMalloc(&c_normal_d, N * N * sizeof(float)));

    // Create events for both kernels
    cudaEvent_t start_tensor, stop_tensor, start_normal, stop_normal;
    CUDA_CHECK(cudaEventCreate(&start_tensor));
    CUDA_CHECK(cudaEventCreate(&stop_tensor));
    CUDA_CHECK(cudaEventCreate(&start_normal));
    CUDA_CHECK(cudaEventCreate(&stop_normal));

    // First run: Tensor Core version
    dim3 gridDim_tensor(N / (WMMA_M * 2), N / (WMMA_N * 2));
    dim3 blockDim_tensor(WARP_SIZE * 2, 2);
    
    printf("\n=== Tensor Core HGEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_tensor));
    hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor>>>(a_d, b_d, c_d);
    CUDA_CHECK(cudaEventRecord(stop_tensor));
    CUDA_CHECK(cudaEventSynchronize(stop_tensor));

    // Second run: Normal version
    dim3 gridDim_normal((N + 31) / 32, (N + 31) / 32);
    dim3 blockDim_normal(32, 32);
    
    printf("\n=== Regular HGEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_normal));
    hgemm_normal<<<gridDim_normal, blockDim_normal>>>(a_d, b_d, c_normal_d);
    CUDA_CHECK(cudaEventRecord(stop_normal));
    CUDA_CHECK(cudaEventSynchronize(stop_normal));

    // Calculate timing for both versions
    float ms_tensor = 0, ms_normal = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_tensor, start_tensor, stop_tensor));
    CUDA_CHECK(cudaEventElapsedTime(&ms_normal, start_normal, stop_normal));

    // Print comparative results
    printf("\n=== Performance Comparison ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Tensor Core kernel time: %.3f ms\n", ms_tensor);
    printf("Regular kernel time: %.3f ms\n", ms_normal);
    printf("Speedup: %.2fx\n", ms_normal / ms_tensor);

    // Verify results (check first element of both)
    float *c_tensor_h = new float[N * N];
    float *c_normal_h = new float[N * N];
    
    CUDA_CHECK(cudaMemcpy(c_tensor_h, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_normal_h, c_normal_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n=== Result Verification ===\n");
    printf("Tensor Core C[0][0] = %.0f\n", c_tensor_h[0]);
    printf("Regular C[0][0] = %.0f\n", c_normal_h[0]);
    printf("Expected value = %d\n", N);

    // Cleanup everything
    delete[] c_tensor_h;
    delete[] c_normal_h;
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;
    
    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(c_d));
    CUDA_CHECK(cudaFree(c_normal_d));
    
    CUDA_CHECK(cudaEventDestroy(start_tensor));
    CUDA_CHECK(cudaEventDestroy(stop_tensor));
    CUDA_CHECK(cudaEventDestroy(start_normal));
    CUDA_CHECK(cudaEventDestroy(stop_normal));

    return 0;
}
