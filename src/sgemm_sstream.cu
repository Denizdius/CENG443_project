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

// CUDA core SGEMM kernel
__global__ void sgemm_cuda_core(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n/2 && col < n) {  // Process top half of matrix
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

    // Calculate position - process bottom half of matrix
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;
    
    // Offset to start from middle of matrix
    warpM += n/(2*WMMA_M);

    fill_fragment(acc_frag, 0.0f);

    if (warpM * WMMA_M < n) {
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
    float *a_h, *b_h, *c_h;
    float *a_d, *b_d, *c_d;
    __half *a_half_d, *b_half_d;
    
    // Create CUDA streams
    cudaStream_t stream_cuda, stream_tensor;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_cuda, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_tensor, cudaStreamNonBlocking));

    // Allocate host memory with pinned memory
    CUDA_CHECK(cudaMallocHost(&a_h, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b_h, N * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&c_h, N * N * sizeof(float)));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a_h[i] = 1.0f;
        b_h[i] = 1.0f;
        c_h[i] = 0.0f;  // Initialize output to zero
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&a_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a_half_d, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b_half_d, N * N * sizeof(__half)));

    // Convert to half precision for Tensor Cores
    __half* a_half_h = new __half[N * N];
    __half* b_half_h = new __half[N * N];
    for (int i = 0; i < N * N; i++) {
        a_half_h[i] = __float2half(a_h[i]);
        b_half_h[i] = __float2half(b_h[i]);
    }

    // Initialize output matrix to zero
    CUDA_CHECK(cudaMemset(c_d, 0, N * N * sizeof(float)));

    // Asynchronous data transfers
    CUDA_CHECK(cudaMemcpyAsync(a_d, a_h, N * N * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_cuda));
    CUDA_CHECK(cudaMemcpyAsync(b_d, b_h, N * N * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_cuda));
    CUDA_CHECK(cudaMemcpyAsync(a_half_d, a_half_h, N * N * sizeof(__half), 
                              cudaMemcpyHostToDevice, stream_tensor));
    CUDA_CHECK(cudaMemcpyAsync(b_half_d, b_half_h, N * N * sizeof(__half), 
                              cudaMemcpyHostToDevice, stream_tensor));

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch configuration
    dim3 grid_tensor((N/2 + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block_tensor(WARP_SIZE, 1);
    dim3 grid_cuda((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N/2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_cuda(BLOCK_SIZE, BLOCK_SIZE);

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernels concurrently on different streams
    // CUDA cores process top half, Tensor cores process bottom half
    sgemm_cuda_core<<<grid_cuda, block_cuda, 0, stream_cuda>>>(a_d, b_d, c_d, N);
    sgemm_tensor_core<<<grid_tensor, block_tensor, 0, stream_tensor>>>(a_half_d, b_half_d, c_d, N);

    // Copy result back
    CUDA_CHECK(cudaMemcpyAsync(c_h, c_d, N * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_cuda));

    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate timing
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Print results
    printf("\n=== Concurrent GEMM Execution ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Total execution time: %.3f ms\n", ms);
    printf("\n=== Result Verification ===\n");
    printf("Top half (CUDA Cores) C[0][0] = %.0f\n", c_h[0]);
    printf("Bottom half (Tensor Cores) C[%d][0] = %.0f\n", N/2, c_h[(N/2)*N]);
    printf("Expected value = %d\n", N);

    // Cleanup
    CUDA_CHECK(cudaFreeHost(a_h));
    CUDA_CHECK(cudaFreeHost(b_h));
    CUDA_CHECK(cudaFreeHost(c_h));
    delete[] a_half_h;
    delete[] b_half_h;

    CUDA_CHECK(cudaFree(a_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(c_d));
    CUDA_CHECK(cudaFree(a_half_d));
    CUDA_CHECK(cudaFree(b_half_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream_cuda));
    CUDA_CHECK(cudaStreamDestroy(stream_tensor));

    return 0;
}
