NVCC = /usr/local/cuda/bin/nvcc
EXEC = CudaExe

SRC = raytracer.cu

SRC_1 = cuda_pattern_td_1.cu
SRC_2 = cuda_pattern_td_2.cu

INC_FOLDER = include
DATA = data

all: build run

build2:
	$(NVCC) -I $(INC_FOLDER) $(SRC_2) -o $(EXEC)2

run2:
	./$(EXEC)2 $(DATA)/input_stencil_1.ppm $(DATA)/output_stencil_2.ppm

build:
	$(NVCC) -I $(INC_FOLDER) $(SRC) -o $(EXEC)

run:
	./$(EXEC)

.PHONY: build
.PHONY: run
