#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include "pixel.hpp"

typedef cv::Vec3b Pixel;

inline BGRPixel BGRfromPixel(const Pixel& pixel) {
    return {static_cast<uchar>(pixel[0]),
            static_cast<uchar>(pixel[1]),
            static_cast<uchar>(pixel[2])};
}

inline BGRPixel* intoBGRPixelArray1D(cv::Mat &image) {
    auto pixelArray = new BGRPixel[image.size().height * image.size().width];

    for (int i = 0; i < image.size().height; i++) {
        for (int j = 0; j < image.size().width; j++) {
            auto &pixel = image.at<Pixel>(i, j);

            pixelArray[i * image.size().width + j] = {static_cast<uchar>(pixel[0]),
                                                      static_cast<uchar>(pixel[1]),
                                                      static_cast<uchar>(pixel[2])};
        }
    }
    return pixelArray;
}

inline cv::Mat fromBGRPixelArray1D(BGRPixel *pixelArray, cv::Size size) {
    auto image = cv::Mat(size, CV_8UC3);
    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            auto &pixel = image.ptr<Pixel>(i)[j];
            pixel[0] = pixelArray[i * size.width + j].data[0];
            pixel[1] = pixelArray[i * size.width + j].data[1];
            pixel[2] = pixelArray[i * size.width + j].data[2];
        }
    }
    return image;
}

inline BGRPixel* allocateBGRPixelArray1D(cv::Size size) {
    return new BGRPixel[size.height * size.width];
}

inline void freeBGRPixelArray1D(BGRPixel *pixelArray) { delete[] pixelArray; }

#endif //BGR_IMAGE_H
