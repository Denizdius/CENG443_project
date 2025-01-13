#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <stdio.h>


#define N_NORMAL 2048
#define N_TENSOR 2048
#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define NUM_STREAMS_PER_TYPE 4  
#define NUM_CHUNKS 16         
#define PIPELINE_STAGES 3     

struct ComputePipeline {
    cudaStream_t stream;
    cudaEvent_t completion;
    half *input_buffer;
    float *output_buffer;
};

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(2); \
    } \
}

struct PinnedMemoryPool {
    half *a_buffer;
    half *b_buffer;
    float *c_buffer;
    size_t pitch_a;
    size_t pitch_b;
    size_t pitch_c;
    
    void allocate(size_t total_size) {  // Changed back to single parameter
        CUDA_CHECK(cudaMallocHost(&a_buffer, total_size));
        CUDA_CHECK(cudaMallocHost(&b_buffer, total_size));
        CUDA_CHECK(cudaMallocHost(&c_buffer, total_size));
    }
    
    void deallocate() {
        CUDA_CHECK(cudaFreeHost(a_buffer));
        CUDA_CHECK(cudaFreeHost(b_buffer));
        CUDA_CHECK(cudaFreeHost(c_buffer));
    }
};


__global__ void hgemm_normal(const half* __restrict__ A, 
                            const half* __restrict__ B, 
                            float* __restrict__ C, 
                            int start_idx, 
                            int chunk_size,
                            size_t pitch_a,
                            size_t pitch_b,
                            size_t pitch_c) {
    __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    row += start_idx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < N_NORMAL; tile += BLOCK_SIZE) {
        if ((row < start_idx + chunk_size) && (threadIdx.x + tile < N_NORMAL)) {
            As[threadIdx.y][threadIdx.x] = *((half*)((char*)A + row * pitch_a) + tile + threadIdx.x);
            Bs[threadIdx.y][threadIdx.x] = *((half*)((char*)B + (tile + threadIdx.y) * pitch_b) + col);
        }
        __syncthreads();
        
        if (row < start_idx + chunk_size && col < N_NORMAL) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
            }
        }
        __syncthreads();
    }
    
    if (row < start_idx + chunk_size && col < N_NORMAL) {
        *((float*)((char*)C + row * pitch_c) + col) = sum;
    }
}

__global__ void hgemm_tensor_core(const half* __restrict__ A, 
                                 const half* __restrict__ B, 
                                 float* __restrict__ C, 
                                 int start_idx, 
                                 int chunk_size) {
    using namespace nvcuda::wmma;
    
    __shared__ half s_a[2][WMMA_M * WMMA_K];
    __shared__ half s_b[2][WMMA_K * WMMA_N];
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y;
    warpM += start_idx / WMMA_M;
    
    if ((warpM * WMMA_M) < (start_idx + chunk_size)) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        
        fill_fragment(acc_frag, 0.0f);
        
        int buf_idx = 0;
        
        if (threadIdx.x < WMMA_M) {
            for (int k = 0; k < WMMA_K; k++) {
                s_a[0][threadIdx.x * WMMA_K + k] = A[(warpM * WMMA_M + threadIdx.x) * N_TENSOR + k];
                s_b[0][k * WMMA_N + threadIdx.x] = B[k * N_TENSOR + warpN * WMMA_N + threadIdx.x];
            }
        }
        __syncthreads();
        
        for (int k = 0; k < N_TENSOR; k += WMMA_K) {
            if (k + WMMA_K < N_TENSOR && threadIdx.x < WMMA_M) {
                for (int i = 0; i < WMMA_K; i++) {
                    s_a[1-buf_idx][threadIdx.x * WMMA_K + i] = 
                        A[(warpM * WMMA_M + threadIdx.x) * N_TENSOR + k + WMMA_K + i];
                    s_b[1-buf_idx][i * WMMA_N + threadIdx.x] = 
                        B[(k + WMMA_K + i) * N_TENSOR + warpN * WMMA_N + threadIdx.x];
                }
            }
            
            load_matrix_sync(a_frag, &s_a[buf_idx][0], WMMA_K);
            load_matrix_sync(b_frag, &s_b[buf_idx][0], WMMA_N);
            
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            
            buf_idx = 1 - buf_idx;
            __syncthreads();
        }
        
        float* c_tile = C + (warpM * WMMA_M) * N_TENSOR + warpN * WMMA_N;
        store_matrix_sync(c_tile, acc_frag, N_TENSOR, mem_row_major);
    }
}

int main() {
    static_assert(N_TENSOR % WMMA_M == 0, "Matrix size must be divisible by WMMA_M");
    static_assert(N_TENSOR % WMMA_N == 0, "Matrix size must be divisible by WMMA_N");
    static_assert(N_TENSOR % WMMA_K == 0, "Matrix size must be divisible by WMMA_K");

    ComputePipeline *normal_pipelines = new ComputePipeline[NUM_STREAMS_PER_TYPE];
    ComputePipeline *tensor_pipelines = new ComputePipeline[NUM_STREAMS_PER_TYPE];
    
    for (int i = 0; i < NUM_STREAMS_PER_TYPE; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&normal_pipelines[i].stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&tensor_pipelines[i].stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&normal_pipelines[i].completion));
        CUDA_CHECK(cudaEventCreate(&tensor_pipelines[i].completion));
    }

    PinnedMemoryPool host_memory;
    host_memory.allocate(std::max(N_NORMAL * N_NORMAL, N_TENSOR * N_TENSOR) * sizeof(half));

    for (int i = 0; i < N_NORMAL * N_NORMAL; i++) {
        host_memory.a_buffer[i] = host_memory.b_buffer[i] = __float2half(1.0f);
    }
    for (int i = 0; i < N_TENSOR * N_TENSOR; i++) {
        host_memory.a_buffer[i] = host_memory.b_buffer[i] = __float2half(1.0f);
    }

    half *d_a_normal, *d_b_normal, *d_a_tensor, *d_b_tensor;
    float *d_c_normal, *d_c_tensor;
    size_t pitch_a_normal, pitch_b_normal, pitch_c_normal;
    size_t pitch_a_tensor, pitch_b_tensor, pitch_c_tensor;
    
    // Allocate 2D memory with pitch for normal matrices
    CUDA_CHECK(cudaMallocPitch(&d_a_normal, &pitch_a_normal, 
                              N_NORMAL * sizeof(half), N_NORMAL));
    CUDA_CHECK(cudaMallocPitch(&d_b_normal, &pitch_b_normal, 
                              N_NORMAL * sizeof(half), N_NORMAL));
    CUDA_CHECK(cudaMallocPitch(&d_c_normal, &pitch_c_normal, 
                              N_NORMAL * sizeof(float), N_NORMAL));
    
    // Allocate 2D memory with pitch for tensor matrices
    CUDA_CHECK(cudaMallocPitch(&d_a_tensor, &pitch_a_tensor, 
                              N_TENSOR * sizeof(half), N_TENSOR));
    CUDA_CHECK(cudaMallocPitch(&d_b_tensor, &pitch_b_tensor, 
                              N_TENSOR * sizeof(half), N_TENSOR));
    CUDA_CHECK(cudaMallocPitch(&d_c_tensor, &pitch_c_tensor, 
                              N_TENSOR * sizeof(float), N_TENSOR));

    // Copy data for normal matrices
    CUDA_CHECK(cudaMemcpy2D(d_a_normal, pitch_a_normal,
                           host_memory.a_buffer, N_NORMAL * sizeof(half),
                           N_NORMAL * sizeof(half), N_NORMAL,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_b_normal, pitch_b_normal,
                           host_memory.b_buffer, N_NORMAL * sizeof(half),
                           N_NORMAL * sizeof(half), N_NORMAL,
                           cudaMemcpyHostToDevice));

    // Copy data for tensor matrices
    CUDA_CHECK(cudaMemcpy2D(d_a_tensor, pitch_a_tensor,
                           host_memory.a_buffer, N_TENSOR * sizeof(half),
                           N_TENSOR * sizeof(half), N_TENSOR,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_b_tensor, pitch_b_tensor,
                           host_memory.b_buffer, N_TENSOR * sizeof(half),
                           N_TENSOR * sizeof(half), N_TENSOR,
                           cudaMemcpyHostToDevice));

    const int chunk_size_normal = (N_NORMAL + NUM_CHUNKS - 1) / NUM_CHUNKS;
    const int chunk_size_tensor = (N_TENSOR + NUM_CHUNKS - 1) / NUM_CHUNKS;
    
    const int aligned_chunk_normal = ((chunk_size_normal + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    const int aligned_chunk_tensor = ((chunk_size_tensor + WMMA_M - 1) / WMMA_M) * WMMA_M;

    dim3 gridDim_normal((N_NORMAL + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (aligned_chunk_normal + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim_normal(BLOCK_SIZE, BLOCK_SIZE);
    
    dim3 gridDim_tensor((aligned_chunk_tensor + WMMA_M - 1) / WMMA_M, 
                        N_TENSOR / WMMA_N);
    dim3 blockDim_tensor(WARP_SIZE, 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    
    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        int pipeline_idx = chunk % NUM_STREAMS_PER_TYPE;
        int offset_normal = chunk * aligned_chunk_normal;
        int offset_tensor = chunk * aligned_chunk_tensor;

        if (chunk < NUM_CHUNKS - 1) {
            int next_offset_normal = (chunk + 1) * aligned_chunk_normal;
            int next_offset_tensor = (chunk + 1) * aligned_chunk_tensor;
            
            cudaMemcpyAsync(d_a_normal + next_offset_normal * N_NORMAL,
                           host_memory.a_buffer + next_offset_normal * N_NORMAL,
                           aligned_chunk_normal * N_NORMAL * sizeof(half),
                           cudaMemcpyHostToDevice,
                           normal_pipelines[pipeline_idx].stream);
                           
            cudaMemcpyAsync(d_a_tensor + next_offset_tensor * N_TENSOR,
                           host_memory.a_buffer + next_offset_tensor * N_TENSOR,
                           aligned_chunk_tensor * N_TENSOR * sizeof(half),
                           cudaMemcpyHostToDevice,
                           tensor_pipelines[pipeline_idx].stream);
        }

        if (offset_normal < N_NORMAL) {
            hgemm_normal<<<gridDim_normal, blockDim_normal, 0, 
                          normal_pipelines[pipeline_idx].stream>>>
                (d_a_normal, d_b_normal, d_c_normal, offset_normal, 
                 aligned_chunk_normal, pitch_a_normal, pitch_b_normal, pitch_c_normal);
        }
        
        if (offset_tensor < N_TENSOR) {
            hgemm_tensor_core<<<gridDim_tensor, blockDim_tensor, 0, 
                               tensor_pipelines[pipeline_idx].stream>>>
                (d_a_tensor, d_b_tensor, d_c_tensor, offset_tensor, aligned_chunk_tensor);
        }

        CUDA_CHECK(cudaEventRecord(normal_pipelines[pipeline_idx].completion,
                                 normal_pipelines[pipeline_idx].stream));
        CUDA_CHECK(cudaEventRecord(tensor_pipelines[pipeline_idx].completion,
                                 tensor_pipelines[pipeline_idx].stream));

        if (pipeline_idx == NUM_STREAMS_PER_TYPE - 1) {
            for (int i = 0; i < NUM_STREAMS_PER_TYPE; i++) {
                CUDA_CHECK(cudaStreamWaitEvent(normal_pipelines[i].stream,
                                             tensor_pipelines[i].completion));
                CUDA_CHECK(cudaStreamWaitEvent(tensor_pipelines[i].stream,
                                             normal_pipelines[i].completion));
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_total = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));

    delete[] normal_pipelines;
    delete[] tensor_pipelines;
    host_memory.deallocate();

    // Cleanup CUDA resources
    CUDA_CHECK(cudaFree(d_a_normal));
    CUDA_CHECK(cudaFree(d_b_normal));
    CUDA_CHECK(cudaFree(d_c_normal));
    CUDA_CHECK(cudaFree(d_a_tensor));
    CUDA_CHECK(cudaFree(d_b_tensor));
    CUDA_CHECK(cudaFree(d_c_tensor));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Total execution time: %f ms\n", ms_total);
    
    return 0;
}
