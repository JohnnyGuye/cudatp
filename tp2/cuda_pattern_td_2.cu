#include "wb.h"

#define wbCheck(stmt) \
 do { \
 cudaError_t err = stmt; \
 if (err != cudaSuccess) { \
 wbLog(ERROR, "Failed to run stmt ", #stmt); \
 wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)); \
 return -1; \
 } \
 } while (0)

#define GPU_F __device__
#define CPU_F __host__
#define GCPU_F GPU_F CPU_F

template< typename T >
GCPU_F T clamp( T val, T mn, T mx ) {
	if( val < mn ) return mn;
	if( val > mx ) return mx;
	return val;
}

GCPU_F int to1DIndex( int i, int j, int k, int width, int depth ) {
	return ((i * width) + j) * depth + k;
}

__global__ void stencil(float *output, float *input, int width, int height, int depth) {
    
	for( int i = 0; i < height; i++ ) {
		for( int j = 0; j < width; j++ ) {
			for( int k = 0; k < depth - 1; k++ ) {
				auto res = input[ to1DIndex( i, j, k - 1, width, depth ) ]
					+ input[ to1DIndex( i, j, k + 1, width, depth ) ]
					+ input[ to1DIndex( i, j - 1, k, width, depth ) ]
					+ input[ to1DIndex( i, j + 1, k, width, depth ) ]
					+ input[ to1DIndex( i - 1, j, k, width, depth ) ]
					+ input[ to1DIndex( i + 1, j, k, width, depth ) ];
				output[ to1DIndex( i, j, k, width, depth ) ] = clamp( res, 0.0f, 255.0f );
			}
		}
	}

}


static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth) {

	stencil<<<1, 1>>>( deviceOutputData, deviceInputData, width, height, depth );

}

int main(int argc, char *argv[]) {
    wbArg_t arg;

    int width;
    int height;
    int depth;

    char *inputFile;
    wbImage_t input;
    wbImage_t output;

    float *hostInputData;
    float *hostOutputData;
    float *deviceInputData;
    float *deviceOutputData;

    arg = wbArg_read(argc, argv);
    inputFile = wbArg_getInputFile(arg, 0);
    input = wbImport(inputFile);

    width = wbImage_getWidth(input);
    height = wbImage_getHeight(input);
    depth = wbImage_getChannels(input);

    output = wbImage_new(width, height, depth);

    hostInputData = wbImage_getData(input);
    hostOutputData = wbImage_getData(output);

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
    cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    
    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);
    wbImage_delete(output);
    wbImage_delete(input);
    return 0;
}

