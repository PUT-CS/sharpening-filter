#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "image.hpp"
#include "pixel.hpp"

#define KERNEL_WIDTH 5
#define KERNEL_HEIGHT 5

constexpr double KERNEL[KERNEL_HEIGHT][KERNEL_WIDTH] =
{
  -1, -1, -1, -1, -1,
  -1,  2,  2,  2, -1,
  -1,  2,  8,  2, -1,
  -1,  2,  2,  2, -1,
  -1, -1, -1, -1, -1,
};

constexpr double FACTOR = 1. / 8.;

void sharpeningFilter(BGRPixel *pixels, BGRPixel *outputImage, int width, int height) {
    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	    
	    double sumR = 0;
	    double sumG = 0;
	    double sumB = 0;
	    
	    for (int i = 0; i < KERNEL_HEIGHT; i++) {
		for (int j = 0; j < KERNEL_WIDTH; j++) {
		    int pixelX = (x + j - KERNEL_WIDTH / 2) % width;
		    int pixelY = y + i - KERNEL_HEIGHT / 2 % height;
		    
		    if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {
			auto pixel = pixels[pixelY * width + pixelX].pixel;
			sumR += pixel.r * KERNEL[i][j];
			sumG += pixel.g * KERNEL[i][j];
			sumB += pixel.b * KERNEL[i][j];
		    }
		}
	    }
	    
	    BGRPixel pixel;
	    pixel.pixel.r = std::min(std::max((int) (sumR * FACTOR), 0), 255);
	    pixel.pixel.g = std::min(std::max((int) (sumG * FACTOR), 0), 255);
	    pixel.pixel.b = std::min(std::max((int) (sumB * FACTOR), 0), 255);
	    
	    outputImage[y * width + x] = pixel;
	}
    }
}


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

    auto start = std::chrono::high_resolution_clock::now();
    
    sharpeningFilter(pixels, outputPixels, size.width, size.height);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << duration.count() << std::endl;
    
    auto outputMat = fromBGRPixelArray1D(outputPixels, size);
    cv::imwrite(outputPath, outputMat);
    
    return 0;
}
