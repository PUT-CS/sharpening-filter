#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "image.hpp"

int main(int argc, char *argv[]) {
    auto inputPath = argv[1];
    auto outputPath = argv[2];
    if (argc < 3) {
        std::cerr << "Usage: sharpening-filter <input_image> <output_image> [--window window_size]"
                  << std::endl;
        return -1;
    }

    cv::Mat bgrImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    auto size = bgrImage.size();

    if (bgrImage.empty()) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return -1;
    }

    //BGRPixel *pixels = intoBGRPixelArray1D(bgrImage);
    BGRPixel *outputPixels = allocateBGRPixelArray1D(size);

    auto outputMat = fromBGRPixelArray1D(outputPixels, size);
    cv::imwrite(outputPath, bgrImage);
    
    return 0;
}
