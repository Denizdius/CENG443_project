#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

// Using namespace for WMMA operations
using namespace nvcuda::wmma;

// Constants for matrix dimensions
#define M 256
#define N 256
#define K 256
#define WARP_SIZE 32

// CUDA core GEMM kernel
__global__ void cudaCoreGemm(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tensor core GEMM kernel using WMMA
__global__ void tensorCoreGemm(float* A, float* B, float* D, int m, int n, int k) {
    // WMMA fragment declarations
    wmma::fragment<matrix_a, 16, 16, 16, float, row_major> a_frag;
    wmma::fragment<matrix_b, 16, 16, 16, float, row_major> b_frag;
    wmma::fragment<accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<accumulator, 16, 16, 16, float> d_frag;
    
    // Calculate tile positions
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warpM < m/16 && warpN < n/16) {
        // Initialize accumulator fragment
        wmma::fill_fragment(c_frag, 0.0f);
        
        // Loop over k dimension
        for (int i = 0; i < k; i += 16) {
            // Load matrices into fragments
            wmma::load_matrix_sync(a_frag, A + warpM * 16 * k + i, k);
            wmma::load_matrix_sync(b_frag, B + i * n + warpN * 16, n);
            
            // Perform matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        // Store result
        wmma::store_matrix_sync(D + warpM * 16 * n + warpN * 16, c_frag, n, wmma::mem_row_major);
    }
}

int main() {
    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_D;
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C = new float[M * N];  // For CUDA cores result
    h_D = new float[M * N];  // For Tensor cores result
    
    // Initialize matrices with some values
    for(int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for(int i = 0; i < K * N; i++) h_B[i] = 2.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_D, M * N * sizeof(float));
    
    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Copy data to device
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream2);
    
    // Launch kernels in different streams
    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    cudaCoreGemm<<<gridDim, blockDim, 0, stream1>>>(d_A, d_B, d_C, M, N, K);
    
    dim3 tensorBlockDim(128, 4);
    dim3 tensorGridDim((M + 64 - 1) / 64, (N + 64 - 1) / 64);
    tensorCoreGemm<<<tensorGridDim, tensorBlockDim, 0, stream2>>>(d_A, d_B, d_D, M, N, K);
    
    // Copy results back to host
    cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    
    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;
    
    return 0;
}