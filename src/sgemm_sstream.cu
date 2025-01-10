#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Constants
#define N 8192
#define CHUNK_SIZE (N/16)  // Split matrix into chunks
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define NUM_STREAMS 16

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

// CUDA core SGEMM kernel for a chunk
__global__ void sgemm_cuda_core(const float* A, const float* B, float* C, 
                               int n, int chunk_start, int chunk_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Adjust indices for the chunk
    row += chunk_start;
    
    if (row < chunk_start + chunk_size && col < n) {
        float sum = 0.0f;
        #pragma unroll 16
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tensor Core SGEMM kernel for a chunk
__global__ void sgemm_tensor_core(const __half* A, const __half* B, float* C, 
                                 int n, int chunk_start, int chunk_size) {
    using namespace nvcuda::wmma;
    
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Adjust warp indices for the chunk
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    warpM += chunk_start / WMMA_M;
    int warpN = blockIdx.y;

    fill_fragment(acc_frag, 0.0f);

    int chunk_end = (chunk_start + chunk_size) / WMMA_M;
    if (warpM < chunk_end) {
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

    // Allocate host memory with pinned memory
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

    // Convert to half precision
    __half* a_half_h = new __half[N * N];
    __half* b_half_h = new __half[N * N];
    for (int i = 0; i < N * N; i++) {
        a_half_h[i] = __float2half(a_h[i]);
        b_half_h[i] = __float2half(b_h[i]);
    }

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a_half_d, a_half_h, N * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_half_d, b_half_h, N * N * sizeof(__half), cudaMemcpyHostToDevice));

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch configuration for chunks
    dim3 grid_tensor((CHUNK_SIZE + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block_tensor(WARP_SIZE, 1);
    dim3 grid_cuda((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_cuda(BLOCK_SIZE, BLOCK_SIZE);

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // Process chunks
    for (int chunk = 0; chunk < N; chunk += CHUNK_SIZE) {
        // Launch both kernels for their respective chunks
        sgemm_tensor_core<<<grid_tensor, block_tensor, 0, streams[0]>>>(
            a_half_d, b_half_d, c_tensor_d, N, chunk, CHUNK_SIZE);
        
        sgemm_cuda_core<<<grid_cuda, block_cuda, 0, streams[1]>>>(
            a_d, b_d, c_cuda_d, N, chunk, CHUNK_SIZE);
    }

    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(c_tensor_h, c_tensor_d, N * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(c_cuda_h, c_cuda_d, N * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[1]));

    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate timing
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Print results
    printf("\n=== Concurrent Chunked GEMM Execution ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Chunk size: %d\n", CHUNK_SIZE);
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
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
