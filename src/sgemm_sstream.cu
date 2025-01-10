#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Constants
#define N 4096
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define NUM_STREAMS 2  // One for Tensor Core, one for CUDA core

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

// CUDA core SGEMM kernel
__global__ void sgemm_cuda_core(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        #pragma unroll 16
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tensor Core SGEMM kernel
__global__ void sgemm_tensor_core(const __half* A, const __half* B, float* C, int n) {
    using namespace nvcuda::wmma;
    
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;

    fill_fragment(acc_frag, 0.0f);

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

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < n && cCol < n) {
        store_matrix_sync(C + cRow * n + cCol, acc_frag, n, mem_row_major);
    }
}

int main() {
    float *a_h, *b_h;
    float *c_tensor_h, *c_cuda_h;
    float *a_d, *b_d;
    float *c_tensor_d, *c_cuda_d;
    __half *a_half_d, *b_half_d;
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate host memory with pinned memory for better transfer performance
    CUDA_CHECK(cudaMallocHost(&a_h, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b_h, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&c_tensor_h, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&c_cuda_h, N * N * sizeof(float)));

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
    CUDA_CHECK(cudaMalloc(&a_half_d, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b_half_d, N * N * sizeof(__half)));

    // Convert to half precision for Tensor Cores
    __half* a_half_h = new __half[N * N];
    __half* b_half_h = new __half[N * N];
    for (int i = 0; i < N * N; i++) {
        a_half_h[i] = __float2half(a_h[i]);
        b_half_h[i] = __float2half(b_h[i]);
    }

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // Stream 0: CUDA core operations
    CUDA_CHECK(cudaMemcpyAsync(a_d, a_h, N * N * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(b_d, b_h, N * N * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[0]));

    // Stream 1: Tensor core operations
    CUDA_CHECK(cudaMemcpyAsync(a_half_d, a_half_h, N * N * sizeof(__half), 
                              cudaMemcpyHostToDevice, streams[1]));
    CUDA_CHECK(cudaMemcpyAsync(b_half_d, b_half_h, N * N * sizeof(__half), 
                              cudaMemcpyHostToDevice, streams[1]));

    // Launch configuration
    dim3 grid_tensor((N + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block_tensor(WARP_SIZE, 1);
    dim3 grid_cuda((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_cuda(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernels in their respective streams
    sgemm_cuda_core<<<grid_cuda, block_cuda, 0, streams[0]>>>(
        a_d, b_d, c_cuda_d, N);
    sgemm_tensor_core<<<grid_tensor, block_tensor, 0, streams[1]>>>(
        a_half_d, b_half_d, c_tensor_d, N);

    // Asynchronous memory transfers back to host
    CUDA_CHECK(cudaMemcpyAsync(c_cuda_h, c_cuda_d, N * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(c_tensor_h, c_tensor_d, N * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[1]));

    // Record stop time after all operations
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate timing
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Print results
    printf("\n=== Concurrent GEMM Execution (Streams) ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Total execution time: %.3f ms\n", ms);
    printf("\n=== Result Verification ===\n");
    printf("Tensor Core C[0][0] = %.0f\n", c_tensor_h[0]);
    printf("CUDA Core C[0][0] = %.0f\n", c_cuda_h[0]);
    printf("Expected value = %d\n", N);

    // Cleanup
    CUDA_CHECK(cudaFreeHost(a_h));
    CUDA_CHECK(cudaFreeHost(b_h));
    CUDA_CHECK(cudaFreeHost(c_tensor_h));
    CUDA_CHECK(cudaFreeHost(c_cuda_h));
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
    
    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
