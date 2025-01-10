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

// WMMA matrix tiles
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

// Add non-Tensor Core version of HGEMM
__global__ void hgemm_normal(const half* A, const half* B, float* C, int start_idx, int chunk_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Adjust indices based on chunk
    row += start_idx;
    
    if (row < start_idx + chunk_size && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += __half2float(A[row * N + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

__global__ void hgemm_tensor_core(const half* A, const half* B, float* C, int start_idx, int chunk_size) {
    // Each warp computes a 16x16 output tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y;

    // Adjust warpM based on chunk
    warpM += start_idx / WMMA_M;

    // Check if this warp should process this chunk
    if (warpM * WMMA_M >= start_idx + chunk_size) {
        return;
    }

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // Load and multiply
    for (int k = 0; k < N; k += WMMA_K) {
        const half* a_tile = A + (warpM * WMMA_M) * N + k;
        const half* b_tile = B + k * N + warpN * WMMA_N;
        
        nvcuda::wmma::load_matrix_sync(a_frag, a_tile, N);
        nvcuda::wmma::load_matrix_sync(b_frag, b_tile, N);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store the output
    float* c_tile = C + (warpM * WMMA_M) * N + warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(c_tile, acc_frag, N, nvcuda::wmma::mem_row_major);
}

int main() {
    // Ensure matrix dimensions are compatible with WMMA
    static_assert(N % WMMA_M == 0, "Matrix size must be divisible by WMMA_M");
    static_assert(N % WMMA_N == 0, "Matrix size must be divisible by WMMA_N");
    static_assert(N % WMMA_K == 0, "Matrix size must be divisible by WMMA_K");

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
    
    // Additional output buffers for concurrent execution
    float *c_stream1_d, *c_stream2_d;
    CUDA_CHECK(cudaMalloc(&c_stream1_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_stream2_d, N * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(half), cudaMemcpyHostToDevice));

    // Create CUDA streams
    cudaStream_t stream_tensor, stream_normal;
    CUDA_CHECK(cudaStreamCreate(&stream_tensor));
    CUDA_CHECK(cudaStreamCreate(&stream_normal));

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Divide the matrix into two parts (ensure it's WMMA aligned)
    const int chunk_size = (N / 2 / WMMA_M) * WMMA_M;

    // Launch configurations
    dim3 gridDim_tensor((chunk_size + WMMA_M - 1) / WMMA_M, N / WMMA_N);
    dim3 blockDim_tensor(WARP_SIZE, 1);

    dim3 gridDim_normal((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim_normal(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim_tensor_full((N + WMMA_M - 1) / WMMA_M, N / WMMA_N);
    dim3 gridDim_normal_full((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float ms_normal = 0, ms_tensor = 0, ms_stream = 0;

    // Clear all output buffers
    CUDA_CHECK(cudaMemset(c_d, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(c_stream1_d, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(c_stream2_d, 0, N * N * sizeof(float)));

    // Test Case 1: Two Tensor Core kernels in different streams
    printf("\n=== Test Case 1: Two Tensor Core Kernels ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor, 0, stream_tensor>>>(a_d, b_d, c_stream1_d, 0, chunk_size);

    hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor, 0, stream_normal>>>(a_d, b_d, c_stream2_d, chunk_size, chunk_size);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_stream, start, stop));
    printf("Two Tensor Core kernels time: %.3f ms\n", ms_stream);

    // Combine results from both streams
    cudaMemcpyAsync(c_d, c_stream1_d, chunk_size * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_tensor);
    cudaMemcpyAsync(c_d + chunk_size * N, c_stream2_d + chunk_size * N, chunk_size * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_normal);
    cudaDeviceSynchronize();

    // Clear buffers for next test
    CUDA_CHECK(cudaMemset(c_stream1_d, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(c_stream2_d, 0, N * N * sizeof(float)));

    // Test Case 2: Two Normal CUDA kernels in different streams
    printf("\n=== Test Case 2: Two Normal CUDA Kernels ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    hgemm_normal<<<gridDim_normal, blockDim_normal, 0, stream_tensor>>>(
        a_d, b_d, c_stream1_d, 0, chunk_size);
        
    hgemm_normal<<<gridDim_normal, blockDim_normal, 0, stream_normal>>>(
        a_d, b_d, c_stream2_d, chunk_size, chunk_size);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_stream, start, stop));
    printf("Two Normal CUDA kernels time: %.3f ms\n", ms_stream);

    // Combine results
    cudaMemcpyAsync(c_d, c_stream1_d, chunk_size * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_tensor);
    cudaMemcpyAsync(c_d + chunk_size * N, c_stream2_d + chunk_size * N, chunk_size * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_normal);
    cudaDeviceSynchronize();

    // Clear buffers for next test
    CUDA_CHECK(cudaMemset(c_stream1_d, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(c_stream2_d, 0, N * N * sizeof(float)));

    // Test Case 3: Mixed Tensor Core and Normal CUDA kernels
    printf("\n=== Test Case 3: Mixed Tensor Core and Normal CUDA Kernels ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor, 0, stream_tensor>>>(
        a_d, b_d, c_stream1_d, 0, chunk_size);
            
    hgemm_normal<<<gridDim_normal, blockDim_normal, 0, stream_normal>>>(
        a_d, b_d, c_stream2_d, chunk_size, chunk_size);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_stream, start, stop));
    printf("Mixed kernels time: %.3f ms\n", ms_stream);

    // Combine results
    cudaMemcpyAsync(c_d, c_stream1_d, chunk_size * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_tensor);
    cudaMemcpyAsync(c_d + chunk_size * N, c_stream2_d + chunk_size * N, chunk_size * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_normal);
    cudaDeviceSynchronize();

    // Print comparative results
    printf("\n=== Performance Comparison ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("1. Normal CUDA cores (full matrix): %.3f ms\n", ms_normal);
    printf("2. Tensor cores (full matrix): %.3f ms\n", ms_tensor);
    printf("3. Streamed version (half tensor + half normal): %.3f ms\n", ms_stream);
    printf("\nSpeedup Analysis:\n");
    printf("Tensor vs Normal: %.2fx\n", ms_normal / ms_tensor);
    printf("Streamed vs Normal: %.2fx\n", ms_normal / ms_stream);
    printf("Streamed vs Tensor: %.2fx\n", ms_tensor / ms_stream);

    // Verify results
    float *c_verify_h = new float[N * N];
    CUDA_CHECK(cudaMemcpy(c_verify_h, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n=== Result Verification ===\n");
    printf("C[0][0] = %.0f\n", c_verify_h[0]);
    printf("C[%d][0] = %.0f\n", chunk_size, c_verify_h[chunk_size * N]);
    printf("Expected value = %d\n", N);

    // Cleanup
    delete[] c_verify_h;
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;
    
    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(c_d));
    CUDA_CHECK(cudaFree(c_stream1_d));
    CUDA_CHECK(cudaFree(c_stream2_d));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream_tensor));
    CUDA_CHECK(cudaStreamDestroy(stream_normal));

    return 0;
}
