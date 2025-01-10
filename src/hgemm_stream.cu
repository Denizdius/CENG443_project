#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <stdio.h>

// Enable Tensor Core operations
#pragma enable_tf32_tensor_core_optimization

// Increased matrix size for better analysis
#define N 4096  // Increased from 1024 to 4096
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define NUM_STREAMS 4  // Using 4 streams for better concurrency

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
    // Calculate warp and matrix positions
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y;
    
    // Adjust warpM based on chunk
    warpM += start_idx / WMMA_M;

    // Ensure all threads in warp participate in WMMA operations
    if ((warpM * WMMA_M) < (start_idx + chunk_size)) {
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
    // Ensure warp synchronization
    __syncwarp();
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

    // Create CUDA streams first
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Copy data to device asynchronously using different streams
    CUDA_CHECK(cudaMemcpyAsync(a_d, a_h, N * N * sizeof(half), cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(b_d, b_h, N * N * sizeof(half), cudaMemcpyHostToDevice, streams[1]));

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Create mid-event for kernel synchronization
    cudaEvent_t mid_event;
    CUDA_CHECK(cudaEventCreate(&mid_event));

    // Divide the matrix into NUM_STREAMS parts
    const int chunk_size = (N / NUM_STREAMS / WMMA_M) * WMMA_M;

    // Launch configurations
    dim3 gridDim_tensor((chunk_size + WMMA_M - 1) / WMMA_M, N / WMMA_N);
    dim3 blockDim_tensor(WARP_SIZE, 1);

    dim3 gridDim_normal((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim_normal(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim_tensor_full((N + WMMA_M - 1) / WMMA_M, N / WMMA_N);
    dim3 gridDim_normal_full((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float ms_normal = 0, ms_tensor = 0, ms_stream = 0;

    // Test 1: All Normal CUDA cores with multiple streams
    /*printf("\n=== Normal CUDA Cores Performance (Full Matrix) ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        hgemm_normal<<<gridDim_normal, blockDim_normal, 0, streams[i]>>>(
            a_d, b_d, c_d, offset, chunk_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_normal, start, stop));

    // Clear output buffer
    CUDA_CHECK(cudaMemset(c_d, 0, N * N * sizeof(float)));*/

    // Test 2: All Tensor cores with multiple streams
    /*printf("\n=== Tensor Cores Performance (Full Matrix) ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor, 0, streams[i]>>>(
            a_d, b_d, c_d, offset, chunk_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_tensor, start, stop));

    // Clear output buffer
    CUDA_CHECK(cudaMemset(c_d, 0, N * N * sizeof(float)));*/

    // Test 3: Mixed Tensor and Normal cores with multiple streams
    printf("\n=== Concurrent HGEMM Performance (Mixed Tensor + Normal) ===\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    // Divide work into smaller chunks for better overlap
    const int num_chunks = 16;  // Increased number of chunks
    const int small_chunk = chunk_size / num_chunks;
    
    // Ensure chunk size is aligned with WMMA tile size
    const int aligned_chunk = (small_chunk + WMMA_M - 1) / WMMA_M * WMMA_M;
    
    // Launch alternating tensor and normal kernels
    for (int i = 0; i < num_chunks; i++) {
        // Launch tensor core kernel for first half
        int tensor_offset = i * aligned_chunk;
        if (tensor_offset < N/2) {
            hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor, 0, streams[0]>>>(
                a_d, b_d, c_d, tensor_offset, aligned_chunk);
        }
        
        // Launch normal kernel for second half
        int normal_offset = N/2 + i * small_chunk;
        if (normal_offset < N) {
            hgemm_normal<<<gridDim_normal, blockDim_normal, 0, streams[1]>>>(
                a_d, b_d, c_d, normal_offset, small_chunk);
        }
        
        // Add small delay between launches to help scheduler
        if (i < num_chunks-1) {
            cudaEventRecord(mid_event, streams[0]);
            cudaStreamWaitEvent(streams[1], mid_event, 0);
        }
    }

    // Copy results back asynchronously
    float *c_verify_h = new float[N * N];
    CUDA_CHECK(cudaMemcpyAsync(c_verify_h, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost, streams[0]));

    // Synchronize all streams before checking results
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_stream, start, stop));

    // Print comparative results
    printf("\n=== Performance Results ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Number of streams: %d\n", NUM_STREAMS);
    printf("Mixed version (concurrent tensor + normal) time: %.3f ms\n", ms_stream);

    // Verify results
    printf("\n=== Result Verification ===\n");
    printf("C[0][0] = %.0f\n", c_verify_h[0]);
    printf("C[%d][0] = %.0f\n", N/2, c_verify_h[(N/2) * N]);
    printf("Expected value = %d\n", N);

    // Cleanup
    delete[] c_verify_h;
    delete[] a_h;
    delete[] b_h;
    delete[] c_h;
    
    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(c_d));
    
    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(mid_event));
    
    // Cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
