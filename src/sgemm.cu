#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Constants
#define N 1024
#define BLOCK_SIZE 16
#define WARP_SIZE 32

// WMMA dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

// Normal SGEMM kernel
__global__ void sgemm_normal(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tensor Core SGEMM kernel
__global__ void sgemm_tensor_core(const __half* A, const __half* B, float* C) {
    using namespace nvcuda::wmma;
    
    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Calculate position
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize accumulator
    fill_fragment(acc_frag, 0.0f);

    // Main loop
    for (int k = 0; k < N; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        load_matrix_sync(a_frag, A + aRow * N + aCol, N);
        load_matrix_sync(b_frag, B + bRow * N + bCol, N);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    store_matrix_sync(C + cRow * N + cCol, acc_frag, N, mem_row_major);
}

int main() {
    float *a_h, *b_h, *c_h;
    float *a_d, *b_d, *c_d;
    __half *a_half_d, *b_half_d;

    // Allocate host memory
    a_h = new float[N * N];
    b_h = new float[N * N];
    c_h = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a_h[i] = 1.0f;
        b_h[i] = 1.0f;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&a_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_d, N * N * sizeof(float)));
    // Convert to __half and copy to device
    __half* a_half_h = new __half[N * N];
    __half* b_half_h = new __half[N * N];
    for (int i = 0; i < N * N; i++) {
        a_half_h[i] = __float2half(a_h[i]);
        b_half_h[i] = __float2half(b_h[i]);
    }
    CUDA_CHECK(cudaMalloc(&a_half_d, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b_half_d, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(a_half_d, a_half_h, N * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_half_d, b_half_h, N * N * sizeof(__half), cudaMemcpyHostToDevice));
    delete[] a_half_h;
    delete[] b_half_h;

    // Copy original float arrays to device
    CUDA_CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate second output array for normal version
    float *c_normal_d;
    CUDA_CHECK(cudaMalloc(&c_normal_d, N * N * sizeof(float)));

    // Setup timing
    cudaEvent_t start_tensor, stop_tensor, start_normal, stop_normal;
    CUDA_CHECK(cudaEventCreate(&start_tensor));
    CUDA_CHECK(cudaEventCreate(&stop_tensor));
    CUDA_CHECK(cudaEventCreate(&start_normal));
    CUDA_CHECK(cudaEventCreate(&stop_normal));

    // Launch Tensor Core kernel
    dim3 grid((N + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block(WARP_SIZE, 1);

    printf("\n=== Tensor Core SGEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_tensor));
    sgemm_tensor_core<<<grid, block>>>(a_half_d, b_half_d, c_d);
    CUDA_CHECK(cudaEventRecord(stop_tensor));
    CUDA_CHECK(cudaEventSynchronize(stop_tensor));

    // Launch normal SGEMM kernel
    dim3 grid_normal((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_normal(BLOCK_SIZE, BLOCK_SIZE);

    printf("\n=== Regular SGEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_normal));
    sgemm_normal<<<grid_normal, block_normal>>>(a_d, b_d, c_normal_d);
    CUDA_CHECK(cudaEventRecord(stop_normal));
    CUDA_CHECK(cudaEventSynchronize(stop_normal));

    // Calculate timing
    float ms_tensor = 0, ms_normal = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_tensor, start_tensor, stop_tensor));
    CUDA_CHECK(cudaEventElapsedTime(&ms_normal, start_normal, stop_normal));

    // Print results
    printf("\n=== Performance Comparison ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Tensor Core kernel time: %.3f ms\n", ms_tensor);
    printf("Regular kernel time: %.3f ms\n", ms_normal);
    printf("Speedup: %.2fx\n", ms_normal / ms_tensor);

    // Verify results
    float *c_tensor_h = new float[N * N];
    float *c_normal_h = new float[N * N];
    
    CUDA_CHECK(cudaMemcpy(c_tensor_h, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_normal_h, c_normal_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n=== Result Verification ===\n");
    printf("Tensor Core C[0][0] = %.0f\n", c_tensor_h[0]);
    printf("Regular C[0][0] = %.0f\n", c_normal_h[0]);
    printf("Expected value = %d\n", N);

    // Cleanup
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