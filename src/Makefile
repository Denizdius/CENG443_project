NVCC = nvcc
NVCCFLAGS = -O3 --use_fast_math -lcudart -lcuda --ptxas-options=-v

TARGETS = hgemm sgemm hgemm_sm80 sgemm_sm80 hgemm_sm75 sgemm_sm75
SRC_HGEMM = hgemm.cu
SRC_SGEMM = sgemm.cu

all: $(TARGETS)

hgemm: $(SRC_HGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -o $@ $<

sgemm: $(SRC_SGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_89 -gencode=arch=compute_89,code=sm_89 -o $@ $<

hgemm_sm80: $(SRC_HGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -o $@ $<

sgemm_sm80: $(SRC_SGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -o $@ $<

hgemm_sm75: $(SRC_HGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -gencode=arch=compute_75,code=sm_75 -o $@ $<

sgemm_sm75: $(SRC_SGEMM)
	$(NVCC) $(NVCCFLAGS) -arch=sm_75 -gencode=arch=compute_75,code=sm_75 -o $@ $<

clean:
	rm -f $(TARGETS)
