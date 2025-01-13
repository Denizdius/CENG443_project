#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <stdio.h>

// Enable Tensor Core operations
#pragma enable_tf32_tensor_core_optimization

// Increased matrix size for better analysis
#define N_NORMAL 2048  // Increased size for normal CUDA cores
#define N_TENSOR 8192  // Size for tensor cores (8x larger)
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define NUM_STREAMS 2  // One for each kernel type
#define NUM_CHUNKS 8  // Increase chunks for better interleaving
#define NUM_STREAMS_PER_TYPE 2  // Keep two streams per type

// WMMA matrix tiles
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", _FILE, __LINE_, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

// Add non-Tensor Core version of HGEMM
_global_ void hgemm_normal(const half* A, const half* B, float* C, int start_idx, int chunk_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Adjust indices based on chunk
    row += start_idx;
    
    if (row < start_idx + chunk_size && col < N_NORMAL) {
        float sum = 0.0f;
        for (int k = 0; k < N_NORMAL; k++) {
            sum += __half2float(A[row * N_NORMAL + k]) * __half2float(B[k * N_NORMAL + col]);
        }
        C[row * N_NORMAL + col] = sum;
    }
}

_global_ void hgemm_tensor_core(const half* A, const half* B, float* C, int start_idx, int chunk_size) {
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
        for (int k = 0; k < N_TENSOR; k += WMMA_K) {
            const half* a_tile = A + (warpM * WMMA_M) * N_TENSOR + k;
            const half* b_tile = B + k * N_TENSOR + warpN * WMMA_N;
            
            nvcuda::wmma::load_matrix_sync(a_frag, a_tile, N_TENSOR);
            nvcuda::wmma::load_matrix_sync(b_frag, b_tile, N_TENSOR);
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        // Store the output
        float* c_tile = C + (warpM * WMMA_M) * N_TENSOR + warpN * WMMA_N;
        nvcuda::wmma::store_matrix_sync(c_tile, acc_frag, N_TENSOR, nvcuda::wmma::mem_row_major);
    }
    // Ensure warp synchronization
    __syncwarp();
}

int main() {
    // Ensure matrix dimensions are compatible with WMMA
    static_assert(N_TENSOR % WMMA_M == 0, "Matrix size must be divisible by WMMA_M");
    static_assert(N_TENSOR % WMMA_N == 0, "Matrix size must be divisible by WMMA_N");
    static_assert(N_TENSOR % WMMA_K == 0, "Matrix size must be divisible by WMMA_K");

    half *a_normal_h, *b_normal_h, *a_tensor_h, *b_tensor_h;
    float *c_normal_h, *c_tensor_h;
    half *a_normal_d, *b_normal_d, *a_tensor_d, *b_tensor_d;
    float *c_normal_d, *c_tensor_d;

    // Allocate host memory
    a_normal_h = new half[N_NORMAL * N_NORMAL];
    b_normal_h = new half[N_NORMAL * N_NORMAL];
    c_normal_h = new float[N_NORMAL * N_NORMAL];
    
    a_tensor_h = new half[N_TENSOR * N_TENSOR];
    b_tensor_h = new half[N_TENSOR * N_TENSOR];
    c_tensor_h = new float[N_TENSOR * N_TENSOR];

    // Initialize matrices
    for (int i = 0; i < N_NORMAL * N_NORMAL; i++) {
        a_normal_h[i] = b_normal_h[i] = __float2half(1.0f);
    }
    for (int i = 0; i < N_TENSOR * N_TENSOR; i++) {
        a_tensor_h[i] = b_tensor_h[i] = __float2half(1.0f);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&a_normal_d, N_NORMAL * N_NORMAL * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&b_normal_d, N_NORMAL * N_NORMAL * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&c_normal_d, N_NORMAL * N_NORMAL * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&a_tensor_d, N_TENSOR * N_TENSOR * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&b_tensor_d, N_TENSOR * N_TENSOR * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&c_tensor_d, N_TENSOR * N_TENSOR * sizeof(float)));

    // Create regular streams instead of priority streams
    cudaStream_t stream_normal[NUM_STREAMS_PER_TYPE];
    cudaStream_t stream_tensor[NUM_STREAMS_PER_TYPE];
    
    for (int i = 0; i < NUM_STREAMS_PER_TYPE; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream_normal[i]));
        CUDA_CHECK(cudaStreamCreate(&stream_tensor[i]));
    }

    // Copy data asynchronously
    CUDA_CHECK(cudaMemcpyAsync(a_normal_d, a_normal_h, N_NORMAL * N_NORMAL * sizeof(half), 
                              cudaMemcpyHostToDevice, stream_normal[0]));
    CUDA_CHECK(cudaMemcpyAsync(b_normal_d, b_normal_h, N_NORMAL * N_NORMAL * sizeof(half), 
                              cudaMemcpyHostToDevice, stream_normal[0]));
    
    CUDA_CHECK(cudaMemcpyAsync(a_tensor_d, a_tensor_h, N_TENSOR * N_TENSOR * sizeof(half), 
                              cudaMemcpyHostToDevice, stream_tensor[0]));
    CUDA_CHECK(cudaMemcpyAsync(b_tensor_d, b_tensor_h, N_TENSOR * N_TENSOR * sizeof(half), 
                              cudaMemcpyHostToDevice, stream_tensor[0]));

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Create mid-event for kernel synchronization
    cudaEvent_t mid_event;
    CUDA_CHECK(cudaEventCreate(&mid_event));

    // Calculate chunk sizes
    const int chunk_size_normal = N_NORMAL / NUM_CHUNKS;
    const int chunk_size_tensor = N_TENSOR / NUM_CHUNKS;

    // Ensure chunk sizes are aligned
    const int aligned_chunk_normal = (chunk_size_normal + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    const int aligned_chunk_tensor = (chunk_size_tensor + WMMA_M - 1) / WMMA_M * WMMA_M;

    // Launch configurations for chunks
    dim3 gridDim_normal((N_NORMAL + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (aligned_chunk_normal + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim_normal(BLOCK_SIZE, BLOCK_SIZE);
    
    dim3 gridDim_tensor((aligned_chunk_tensor + WMMA_M - 1) / WMMA_M, N_TENSOR / WMMA_N);
    dim3 blockDim_tensor(WARP_SIZE, 1);

    float ms_stream = 0;

    // Test 3: Mixed Tensor and Normal cores with multiple streams
    printf("\n=== Concurrent HGEMM Performance (Mixed Tensor + Normal) ===\n");
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernels without forced synchronization
    for (int i = 0; i < NUM_CHUNKS; i++) {
        int stream_idx = i % NUM_STREAMS_PER_TYPE;
        int offset_normal = i * aligned_chunk_normal;
        int offset_tensor = i * aligned_chunk_tensor;

        // Launch both kernels without synchronization between them
        if (offset_normal < N_NORMAL) {
            hgemm_normal<<<gridDim_normal, blockDim_normal, 0, stream_normal[stream_idx]>>>(
                a_normal_d, b_normal_d, c_normal_d, offset_normal, aligned_chunk_normal);
        }
        
        if (offset_tensor < N_TENSOR) {
            hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor, 0, stream_tensor[stream_idx]>>>(
                a_tensor_d, b_tensor_d, c_tensor_d, offset_tensor, aligned_chunk_tensor);
        }

        // Add minimal synchronization only between chunks if needed
        if (i < NUM_CHUNKS - 1) {
            CUDA_CHECK(cudaEventRecord(mid_event, stream_tensor[stream_idx]));
            cudaStreamWaitEvent(stream_normal[(stream_idx + 1) % NUM_STREAMS_PER_TYPE], mid_event);
        }
    }

    // Copy results back asynchronously
    float *c_verify_h = new float[N_TENSOR * N_TENSOR];
    CUDA_CHECK(cudaMemcpyAsync(c_verify_h, c_tensor_d, N_TENSOR * N_TENSOR * sizeof(float), cudaMemcpyDeviceToHost, stream_tensor[0]));

    // Synchronize all streams before checking results
    for (int i = 0; i < NUM_STREAMS_PER_TYPE; i++) {
        CUDA_CHECK(cudaStreamSynchronize(stream_normal[i]));
        CUDA_CHECK(cudaStreamSynchronize(stream_tensor[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_stream, start, stop));

    // Print comparative results
    printf("\n=== Performance Results ===\n");
    printf("Matrix size (Normal): %dx%d\n", N_NORMAL, N_NORMAL);
    printf("Matrix size (Tensor): %dx%d\n", N_TENSOR, N_TENSOR);
    printf("Number of streams: %d\n", NUM_STREAMS);
    printf("Mixed version (concurrent tensor + normal) time: %.3f ms\n", ms_stream);

    // Verify results
    printf("\n=== Result Verification ===\n");
    printf("C[0][0] = %.0f\n", c_verify_h[0]);
    printf("C[%d][0] = %.0f\n", N_TENSOR/2, c_verify_h[(N_TENSOR/2) * N_TENSOR]);
    printf("Expected value = %d\n", N_TENSOR);

    // Cleanup
    delete[] c_verify_h;
    delete[] a_normal_h;
    delete[] b_normal_h;
    delete[] c_normal_h;
    delete[] a_tensor_h;
    delete[] b_tensor_h;
    delete[] c_tensor_h;
    
    CUDA_CHECK(cudaFree(a_normal_d));
    CUDA_CHECK(cudaFree(b_normal_d));
    CUDA_CHECK(cudaFree(c_normal_d));
    CUDA_CHECK(cudaFree(a_tensor_d));
    CUDA_CHECK(cudaFree(b_tensor_d));
    CUDA_CHECK(cudaFree(c_tensor_d));
    
    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(mid_event));
    
    // Update stream cleanup
    for (int i = 0; i < NUM_STREAMS_PER_TYPE; i++) {
        CUDA_CHECK(cudaStreamDestroy(stream_normal[i]));
        CUDA_CHECK(cudaStreamDestroy(stream_tensor[i]));
    }

    return 0;
}