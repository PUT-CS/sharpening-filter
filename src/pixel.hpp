#ifndef PIXEL_HPP
#define PIXEL_HPP

typedef unsigned char uchar;
typedef uchar PixelValue;

typedef union {
  struct BGRPixel {
    uchar b;
    uchar g;
    uchar r;
  } pixel;
  uchar data[3];
} BGRPixel;

#endif
