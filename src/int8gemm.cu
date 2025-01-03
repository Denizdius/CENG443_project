#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

// Constants
#define N 1024
#define BLOCK_SIZE 16
#define WARP_SIZE 32

// WMMA dimensions for INT8
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

// Regular CUDA Core INT8 GEMM implementation
__global__ void int8gemm_normal(const int8_t* A, const int8_t* B, int32_t* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int32_t sum = 0;
        for (int k = 0; k < N; k++) {
            sum += static_cast<int32_t>(A[row * N + k]) * 
                   static_cast<int32_t>(B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

// Tensor Core INT8 GEMM implementation
__global__ void int8gemm_tensor_core(const int8_t* A, const int8_t* B, int32_t* C) {
    using namespace nvcuda;
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;

    // Calculate block position
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize accumulator with zeros
    wmma::fill_fragment(acc_frag, 0);

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
    int8_t *a_h, *b_h;
    int32_t *c_h;
    int8_t *a_d, *b_d;
    int32_t *c_d;

    // Allocate host memory
    a_h = new int8_t[N * N];
    b_h = new int8_t[N * N];
    c_h = new int32_t[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a_h[i] = 1;
        b_h[i] = 1;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&a_d, N * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&b_d, N * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&c_d, N * N * sizeof(int32_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Allocate second output array for normal version
    int32_t *c_normal_d;
    CUDA_CHECK(cudaMalloc(&c_normal_d, N * N * sizeof(int32_t)));

    // Setup timing
    cudaEvent_t start_tensor, stop_tensor, start_normal, stop_normal;
    CUDA_CHECK(cudaEventCreate(&start_tensor));
    CUDA_CHECK(cudaEventCreate(&stop_tensor));
    CUDA_CHECK(cudaEventCreate(&start_normal));
    CUDA_CHECK(cudaEventCreate(&stop_normal));

    // Launch Tensor Core kernel
    dim3 grid((N + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block(WARP_SIZE, 1);

    printf("\n=== Tensor Core INT8 GEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_tensor));
    int8gemm_tensor_core<<<grid, block>>>(a_d, b_d, c_d);
    CUDA_CHECK(cudaEventRecord(stop_tensor));
    CUDA_CHECK(cudaEventSynchronize(stop_tensor));

    // Launch normal INT8 GEMM kernel
    dim3 grid_normal((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_normal(BLOCK_SIZE, BLOCK_SIZE);

    printf("\n=== Regular INT8 GEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_normal));
    int8gemm_normal<<<grid_normal, block_normal>>>(a_d, b_d, c_normal_d);
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
    int32_t *c_tensor_h = new int32_t[N * N];
    int32_t *c_normal_h = new int32_t[N * N];
    
    CUDA_CHECK(cudaMemcpy(c_tensor_h, c_d, N * N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_normal_h, c_normal_d, N * N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    
    printf("\n=== Result Verification ===\n");
    printf("Tensor Core C[0][0] = %d\n", c_tensor_h[0]);
    printf("Regular C[0][0] = %d\n", c_normal_h[0]);
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