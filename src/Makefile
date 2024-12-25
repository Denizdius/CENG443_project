NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89 --use_fast_math -lcudart -lcuda --ptxas-options=-v \
            -gencode=arch=compute_89,code=sm_89

TARGET = hgemm
SRC = hgemm.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
