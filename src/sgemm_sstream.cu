#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Constants for matrix dimensions
#define M 4096
#define N 4096
#define K 4096
#define WARP_SIZE 32

// CUDA core GEMM kernel
__global__ void cudaCoreGemm(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m/2 && col < n) {  // Process first half of matrix
        float sum = 0.0f;
        #pragma unroll 16
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tensor core GEMM kernel using WMMA with FP32
__global__ void tensorCoreGemm(const float* A, const float* B, float* C, int m, int n, int k) {
    // WMMA fragment declarations
    using namespace nvcuda::wmma;
    
    // Each thread block handles a 16x16x16 matrix multiplication
    fragment<matrix_a, 16, 16, 16, float, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, float, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;
    
    // Calculate tile positions
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y;
    
    // Offset to process second half of matrix
    warpM += m/(2*16);
    
    if (warpM < m/16 && warpN < n/16) {
        // Initialize accumulator fragment
        fill_fragment(acc_frag, 0.0f);
        
        // Loop over k dimension
        for (int i = 0; i < k; i += 16) {
            int aRow = warpM * 16;
            int aCol = i;
            int bRow = i;
            int bCol = warpN * 16;

            if (aRow < m && aCol < k && bRow < k && bCol < n) {
                // Load matrices into fragments
                load_matrix_sync(a_frag, A + aRow * k + aCol, k);
                load_matrix_sync(b_frag, B + bRow * n + bCol, n);
                
                // Perform matrix multiplication
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }
        
        // Store result
        int cRow = warpM * 16;
        int cCol = warpN * 16;
        if (cRow < m && cCol < n) {
            store_matrix_sync(C + cRow * n + cCol, acc_frag, n, mem_row_major);
        }
    }
}

int main() {
    // Allocate host memory
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_C, M * N * sizeof(float)));
    
    // Initialize matrices with some values
    for(int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for(int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for(int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Create CUDA streams with priorities
    cudaStream_t stream_cuda, stream_tensor;
    int priority_high, priority_low;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_tensor, cudaStreamNonBlocking, priority_high));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_cuda, cudaStreamNonBlocking, priority_low));
    
    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start time
    CUDA_CHECK(cudaEventRecord(start));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream_cuda));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream_cuda));
    
    // Launch kernels in different streams
    // CUDA cores process first half, Tensor cores process second half
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M/2 + blockDim.y - 1) / blockDim.y);
    cudaCoreGemm<<<gridDim, blockDim, 0, stream_cuda>>>(d_A, d_B, d_C, M, N, K);
    
    dim3 tensorBlockDim(WARP_SIZE, 1);
    dim3 tensorGridDim((M/2 + 16 - 1) / 16, (N + 16 - 1) / 16);
    tensorCoreGemm<<<tensorGridDim, tensorBlockDim, 0, stream_tensor>>>(d_A, d_B, d_C, M, N, K);
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream_cuda));
    
    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate timing
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Print results
    printf("\n=== Concurrent GEMM Execution (FP32) ===\n");
    printf("Matrix size: %dx%d\n", M, N);
    printf("Total execution time: %.3f ms\n", ms);
    printf("\n=== Result Verification ===\n");
    printf("Top half (CUDA Cores) C[0][0] = %.0f\n", h_C[0]);
    printf("Bottom half (Tensor Cores) C[%d][0] = %.0f\n", M/2, h_C[(M/2)*N]);
    printf("Expected value = %d\n", K);
    
    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream_cuda));
    CUDA_CHECK(cudaStreamDestroy(stream_tensor));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    
    return 0;
}