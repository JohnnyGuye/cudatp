NVCC = /usr/local/cuda/bin/nvcc
EXEC_1 = CudaExe1
EXEC_2 = CudaExe2
SRC_1 = cuda_td_1.cu
SRC_2 = cuda_td_2.cu

INC_FOLDER = include

all: build run

run1:
	./$(EXEC_1) dataset/vector/input0.raw dataset/vector/input1.raw dataset/vector/output.raw

run2:
	./$(EXEC_2) dataset/images/logo-insa-lyon.ppm dataset/images/logo-insa-lyon-grey.ppm

run: run1 run2

build1:
	$(NVCC) $(SRC_1) -I $(INC_FOLDER) -o $(EXEC_1)

build2:
	$(NVCC) $(SRC_2) -I $(INC_FOLDER) -o $(EXEC_2)

build: build1 build2
