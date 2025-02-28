#include <wb.h>


//#define wbCheck(stmt)
//    do {
//        cudaError_t err = stmt;
//        if (err != cudaSuccess) {
//            wbLog(ERROR, "Failed to run stmt ", #stmt);
//            wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));
//            return -1;
//        }
//    } while (0)

__global__ void greyscale( float * inputData, float * outputData, int pixelsCount ) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if( i >= pixelsCount ) return;

    i *= 3;

    outputData[ i ] = (
                      inputData[ i ] * 0.21f
                    + inputData[ i + 1 ] * 0.71f
                    + inputData[ i + 2 ] * 0.07f
                ) / 3;

}

//@@ INSERT CODE HERE
int main(int argc, char *argv[]) {
    wbArg_t args;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    char *outputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
    args = wbArg_read(argc, argv); /* parse the input arguments */
    inputImageFile = wbArg_getInputFile(args, 0);
    outputImageFile = wbArg_getInputFile(args, 1);
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    // For this lab the value is always 3
    imageChannels = wbImage_getChannels(inputImage);
    // Since the image is monochromatic, it only contains one channel
    outputImage = wbImage_new(imageWidth, imageHeight, 1);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputImageData,
    imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData,
    imageWidth * imageHeight * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData,
    imageWidth * imageHeight * imageChannels * sizeof(float),
    cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
    ///////////////////////////////////////////////////////
    wbTime_start(Compute, "Doing the computation on the GPU");

    const int THREADS = 256;
    const int BLOCKS = ceil( (float)imageWidth * imageHeight / THREADS );

    greyscale<<<BLOCKS, THREADS>>>(
                                  deviceInputImageData,
                                  deviceOutputImageData,
                                  imageWidth * imageHeight
                                  );

    wbTime_stop(Compute, "Doing the computation on the GPU");
    ///////////////////////////////////////////////////////
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
    imageWidth * imageHeight * sizeof(float),
    cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    
    
    unsigned char grayScale[imageHeight][imageWidth];
 
    int i, j;
    for (j = 0; j < imageHeight; ++j) {
        for (i = 0; i < imageWidth; ++i) {
            grayScale[j][i] = ceil(hostOutputImageData[i + j * imageWidth] * 255.0);
        }
    }
 
    FILE *fp = fopen(outputImageFile, "wb"); /* b - binary mode */
    fprintf(fp, "P5\n%d %d\n255\n", imageWidth, imageHeight);
    fwrite(grayScale, sizeof(grayScale), 1, fp);
    fclose(fp);
    
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    return 0;
}
