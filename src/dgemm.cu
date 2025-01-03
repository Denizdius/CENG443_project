#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <stdio.h>

// Constants
#define N 1024
#define BLOCK_SIZE 16
#define WARP_SIZE 32

// WMMA dimensions for double precision
const int WMMA_M = 8;  // Note: smaller tiles for double precision
const int WMMA_N = 8;
const int WMMA_K = 4;

// Error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

// Regular CUDA Core DGEMM implementation
__global__ void dgemm_normal(const double* A, const double* B, double* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tensor Core DGEMM implementation
__global__ void dgemm_tensor_core(const double* A, const double* B, double* C) {
    using namespace nvcuda;
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> acc_frag;

    // Calculate block position
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize accumulator with zeros
    wmma::fill_fragment(acc_frag, 0.0);

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
    double *a_h, *b_h, *c_h;
    double *a_d, *b_d, *c_d;

    // Allocate host memory
    a_h = new double[N * N];
    b_h = new double[N * N];
    c_h = new double[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        a_h[i] = 1.0;
        b_h[i] = 1.0;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&a_d, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&b_d, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c_d, N * N * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(a_d, a_h, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b_h, N * N * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate second output array for normal version
    double *c_normal_d;
    CUDA_CHECK(cudaMalloc(&c_normal_d, N * N * sizeof(double)));

    // Setup timing
    cudaEvent_t start_tensor, stop_tensor, start_normal, stop_normal;
    CUDA_CHECK(cudaEventCreate(&start_tensor));
    CUDA_CHECK(cudaEventCreate(&stop_tensor));
    CUDA_CHECK(cudaEventCreate(&start_normal));
    CUDA_CHECK(cudaEventCreate(&stop_normal));

    // Launch Tensor Core kernel
    dim3 grid((N + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    dim3 block(WARP_SIZE, 1);

    printf("\n=== Tensor Core DGEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_tensor));
    dgemm_tensor_core<<<grid, block>>>(a_d, b_d, c_d);
    CUDA_CHECK(cudaEventRecord(stop_tensor));
    CUDA_CHECK(cudaEventSynchronize(stop_tensor));

    // Launch normal DGEMM kernel
    dim3 grid_normal((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_normal(BLOCK_SIZE, BLOCK_SIZE);

    printf("\n=== Regular DGEMM Performance ===\n");
    CUDA_CHECK(cudaEventRecord(start_normal));
    dgemm_normal<<<grid_normal, block_normal>>>(a_d, b_d, c_normal_d);
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
    double *c_tensor_h = new double[N * N];
    double *c_normal_h = new double[N * N];
    
    CUDA_CHECK(cudaMemcpy(c_tensor_h, c_d, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_normal_h, c_normal_d, N * N * sizeof(double), cudaMemcpyDeviceToHost));
    
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
