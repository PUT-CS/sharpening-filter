#include "pixel.cuh"
#include "math.h"

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5

__device__ constexpr double KERNEL[KERNEL_HEIGHT][KERNEL_WIDTH] =
{
  -1, -1, -1, -1, -1,
  -1,  2,  2,  2, -1,
  -1,  2,  8,  2, -1,
  -1,  2,  2,  2, -1,
  -1, -1, -1, -1, -1,
};

constexpr double FACTOR = 1. / 8.;

__global__ void sharpeningFilter(BGRPixel *pixels, BGRPixel *outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
	return;
    }
    
    double sumR = 0;
    double sumG = 0;
    double sumB = 0;
    
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
	for (int j = 0; j < KERNEL_WIDTH; j++) {
	    int pixelX = (x + j - KERNEL_WIDTH / 2) % width;
	    int pixelY = (y + i - KERNEL_HEIGHT / 2) % height;
	    
	    if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
		auto pixel = pixels[pixelY * width + pixelX].pixel;
		sumR += pixel.r * KERNEL[i][j];
		sumG += pixel.g * KERNEL[i][j];
		sumB += pixel.b * KERNEL[i][j];
	    }
	}
    }
    
    BGRPixel pixel;
    pixel.pixel.r = min(max((int) (sumR * FACTOR), 0), 255);
    pixel.pixel.g = min(max((int) (sumG * FACTOR), 0), 255);
    pixel.pixel.b = min(max((int) (sumB * FACTOR), 0), 255);
    
    outputImage[y * width + x] = pixel;
}

namespace Kernel {
    void run(BGRPixel *pixels, BGRPixel *outputImage, int width, int height) {
	const int blockX = 16;
	const int blockY = 16;
	dim3 numberOfBlocks(width / blockX +1, height / blockY +1);
	dim3 numberOfThreads(blockX, blockY);
	sharpeningFilter<<<numberOfBlocks, numberOfThreads>>>(pixels, outputImage, width, height);
    }
}
