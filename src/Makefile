NVCC = nvcc
NVCCFLAGS = -O3 --use_fast_math -lcudart -lcuda --ptxas-options=-v

TARGETS = hgemm sgemm dgemm int8gemm hgemm_sm80 sgemm_sm80 dgemm_sm80 int8gemm_sm80 hgemm_sm75 sgemm_sm75 dgemm_sm75 int8gemm_sm75
SRC_HGEMM = hgemm.cu
SRC_SGEMM = sgemm.cu
SRC_DGEMM = dgemm.cu
SRC_INT8GEMM = int8gemm.cu

all: $(TARGETS)

hgemm: $(SRC_HGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -o $@ $<

sgemm: $(SRC_SGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -o $@ $<

dgemm: $(SRC_DGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -o $@ $<

hgemm_sm80: $(SRC_HGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -o $@ $<

sgemm_sm80: $(SRC_SGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -o $@ $<

dgemm_sm80: $(SRC_DGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -o $@ $<

hgemm_sm75: $(SRC_HGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -gencode=arch=compute_75,code=sm_75 -o $@ $<

sgemm_sm75: $(SRC_SGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -gencode=arch=compute_75,code=sm_75 -o $@ $<

dgemm_sm75: $(SRC_DGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -gencode=arch=compute_75,code=sm_75 -o $@ $<

int8gemm: $(SRC_INT8GEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -o $@ $<

int8gemm_sm80: $(SRC_INT8GEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -o $@ $<

int8gemm_sm75: $(SRC_INT8GEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -gencode=arch=compute_75,code=sm_75 -o $@ $<

clean:
	rm -f $(TARGETS)
