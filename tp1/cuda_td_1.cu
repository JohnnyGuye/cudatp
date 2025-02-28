#include "wb.h"
#include <iostream>

#define FLOAT_SIZE sizeof(float)

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if( i >= len ) return;

    out[i] = in1[i] + in2[i];

}


int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 =
    (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 =
    (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The input length is ", inputLength);


    // Memory allocation
    wbTime_start(GPU, "Allocating GPU memory.");

    cudaMalloc((void **) &deviceInput1, inputLength * FLOAT_SIZE);
    cudaMalloc((void **) &deviceInput2, inputLength * FLOAT_SIZE);
    cudaMalloc((void **) &deviceOutput, inputLength * FLOAT_SIZE);

    wbTime_stop(GPU, "Allocating GPU memory.");

    // Memory copy to device
    wbTime_start(GPU, "Copying input memory to the GPU.");

    cudaMemcpy( deviceInput1, hostInput1, inputLength * FLOAT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy( deviceInput2, hostInput2, inputLength * FLOAT_SIZE, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    const int THREADS = 256;
    const int BLOCKS = ceil( (float)inputLength / THREADS );

    wbTime_start(Compute, "Performing CUDA computation");

    vecAdd<<<BLOCKS, THREADS>>>(
                                  deviceInput1,
                                  deviceInput2,
                                  deviceOutput,
                                  inputLength
                                  );

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    // Memory copy to host
    wbTime_start(Copy, "Copying output memory to the CPU");

    cudaMemcpy( hostOutput, deviceOutput, inputLength * FLOAT_SIZE, cudaMemcpyDeviceToHost );

    wbTime_stop(Copy, "Copying output memory to the CPU");
    wbTime_start(GPU, "Freeing GPU Memory");

    cudaFree( deviceInput1 );
    cudaFree( deviceInput2 );
    cudaFree( deviceOutput );

    wbTime_stop(GPU, "Freeing GPU Memory");
    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
