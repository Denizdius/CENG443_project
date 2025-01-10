#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Constants
#define N 4096
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

// Normal CUDA core SGEMM kernel
__global__ void sgemm_cuda_core(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tensor Core SGEMM kernel
__global__ void sgemm_tensor_core(const __half* A, const __half* B, float* C, int n) {
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
    for (int k = 0; k < n; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < n && aCol < n && bRow < n && bCol < n) {
            load_matrix_sync(a_frag, A + aRow * n + aCol, n);
            load_matrix_sync(b_frag, B + bRow * n + bCol, n);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < n && cCol < n) {
        store_matrix_sync(C + cRow * n + cCol, acc_frag, n, mem_row_major);
    }
}

int main() {
    float *a_h, *b_h, *c_tensor_h, *c_cuda_h;
    float *a_d, *b_d, *c_tensor_d, *c_cuda_d;
    __half *a_half_d, *b_half_d;
    
    // Create CUDA streams for concurrent execution
    cudaStream_t stream_tensor, stream_cuda;
    CUDA_CHECK(cudaStreamCreate(&stream_tensor));
    CUDA_CHECK(cudaStreamCreate(&stream_cuda));

    // Allocate host memory
    a_h = new float[N * N];
    b_h = new float[N * N];
    c_tensor_h = new float[N * N];
    c_cuda_h = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a_h[i] = 1.0f;
        b_h[i] = 1.0f;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&a_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_tensor_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_cuda_d, N * N * sizeof(float)));

    // Allocate and convert to half precision for Tensor Cores
    __half* a_half_h = new __half[N * N];
    __half* b_half_h = new __half[N * N];
    for (int i = 0; i < N * N; i++) {
        a_half_h[i] = __float2half(a_h[i]);
        b_half_h[i] = __float2half(b_h[i]);
    }
    
    CUDA_CHECK(cudaMalloc(&a_half_d, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b_half_d, N * N * sizeof(__half)));

    // Asynchronous memory transfers using streams
    CUDA_CHECK(cudaMemcpyAsync(a_half_d, a_half_h, N * N * sizeof(__half), 
                              cudaMemcpyHostToDevice, stream_tensor));
    CUDA_CHECK(cudaMemcpyAsync(b_half_d, b_half_h, N * N * sizeof(__half), 
                              cudaMemcpyHostToDevice, stream_tensor));
    CUDA_CHECK(cudaMemcpyAsync(a_d, a_h, N * N * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_cuda));
    CUDA_CHECK(cudaMemcpyAsync(b_d, b_h, N * N * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_cuda));

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch configuration for Tensor Core kernel
    dim3 grid_tensor((N + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block_tensor(WARP_SIZE, 1);

    // Launch configuration for CUDA core kernel
    dim3 grid_cuda((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_cuda(BLOCK_SIZE, BLOCK_SIZE);

    printf("\n=== Concurrent GEMM Execution ===\n");
    CUDA_CHECK(cudaEventRecord(start));

    // Launch both kernels concurrently in different streams
    sgemm_tensor_core<<<grid_tensor, block_tensor, 0, stream_tensor>>>(
        a_half_d, b_half_d, c_tensor_d, N);
    sgemm_cuda_core<<<grid_cuda, block_cuda, 0, stream_cuda>>>(
        a_d, b_d, c_cuda_d, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate timing
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Asynchronous copy results back
    CUDA_CHECK(cudaMemcpyAsync(c_tensor_h, c_tensor_d, N * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_tensor));
    CUDA_CHECK(cudaMemcpyAsync(c_cuda_h, c_cuda_d, N * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_cuda));

    // Synchronize streams
    CUDA_CHECK(cudaStreamSynchronize(stream_tensor));
    CUDA_CHECK(cudaStreamSynchronize(stream_cuda));

    // Print results
    printf("Matrix size: %dx%d\n", N, N);
    printf("Total execution time: %.3f ms\n", ms);
    printf("\n=== Result Verification ===\n");
    printf("Tensor Core C[0][0] = %.0f\n", c_tensor_h[0]);
    printf("CUDA Core C[0][0] = %.0f\n", c_cuda_h[0]);
    printf("Expected value = %d\n", N);

    // Cleanup
    delete[] a_h;
    delete[] b_h;
    delete[] c_tensor_h;
    delete[] c_cuda_h;
    delete[] a_half_h;
    delete[] b_half_h;

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(c_tensor_d));
    CUDA_CHECK(cudaFree(c_cuda_d));
    CUDA_CHECK(cudaFree(a_half_d));
    CUDA_CHECK(cudaFree(b_half_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream_tensor));
    CUDA_CHECK(cudaStreamDestroy(stream_cuda));

    return 0;
}
