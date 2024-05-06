#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "image.hpp"
#include "pixel.cuh"
#include "kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(int argc, char *argv[]) {
    auto inputPath = argv[1];
    auto outputPath = argv[2];
    if (argc < 3) {
        std::cerr << "Usage: sharpening-filter <input_image> <output_image>"
                  << std::endl;
        return -1;
    }

    cv::Mat bgrImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    auto size = bgrImage.size();

    if (bgrImage.empty()) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return -1;
    }
    
    BGRPixel *pixels = intoBGRPixelArray1D(bgrImage);
    BGRPixel *outputPixels = allocateBGRPixelArray1D(size);

    BGRPixel* devicePixels;
    BGRPixel* deviceOutputPixels; 

    int imageSize = size.width * size.height * sizeof(BGRPixel);
    
    // allocate to the pointers
    cudaMalloc((void**) &devicePixels, imageSize);
    cudaMalloc((void**) &deviceOutputPixels, imageSize);

    // copy image data to the device
    cudaMemcpy(devicePixels, pixels, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutputPixels, outputPixels, imageSize, cudaMemcpyHostToDevice);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Kernel::run(devicePixels, deviceOutputPixels, size.width, size.height);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << duration.count() << std::endl;

    cudaMemcpy(outputPixels, deviceOutputPixels, imageSize, cudaMemcpyDeviceToHost);
    
    auto outputMat = fromBGRPixelArray1D(outputPixels, size);
    cv::imwrite(outputPath, outputMat);
    
    return 0;
}
