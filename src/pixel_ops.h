// Copyright 2026 Adhiraj Singh
//
// Per-pixel image operations shared by the CUDA kernels and the CPU
// reference implementation. Keeping the math in one place guarantees that
// the GPU and CPU paths compute the same thing, which is what makes the
// --verify option meaningful.

#ifndef SRC_PIXEL_OPS_H_
#define SRC_PIXEL_OPS_H_

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define EDGE_HOST_DEVICE __host__ __device__
#else
#define EDGE_HOST_DEVICE
#endif

namespace edge {

// Largest supported Gaussian radius (kernel width = 2 * radius + 1).
constexpr int kMaxGaussianRadius = 15;
constexpr int kMaxGaussianTaps = 2 * kMaxGaussianRadius + 1;

// Clamps an index into [0, size - 1] (clamp-to-edge border handling).
EDGE_HOST_DEVICE inline int ClampIndex(int i, int size) {
  return i < 0 ? 0 : (i >= size ? size - 1 : i);
}

// ITU-R BT.601 luma from 8-bit RGB.
EDGE_HOST_DEVICE inline float GrayFromRgb(uint8_t r, uint8_t g, uint8_t b) {
  return 0.299f * r + 0.587f * g + 0.114f * b;
}

// One output pixel of the horizontal pass of a separable Gaussian blur.
EDGE_HOST_DEVICE inline float GaussianRowAt(const float* in, int width,
                                            int x, int y,
                                            const float* weights,
                                            int radius) {
  float sum = 0.0f;
  const float* row = in + static_cast<int64_t>(y) * width;
  for (int k = -radius; k <= radius; ++k) {
    sum += weights[k + radius] * row[ClampIndex(x + k, width)];
  }
  return sum;
}

// One output pixel of the vertical pass of a separable Gaussian blur.
EDGE_HOST_DEVICE inline float GaussianColumnAt(const float* in, int width,
                                               int height, int x, int y,
                                               const float* weights,
                                               int radius) {
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum += weights[k + radius] *
           in[static_cast<int64_t>(ClampIndex(y + k, height)) * width + x];
  }
  return sum;
}

// Sobel gradient magnitude at (x, y).
EDGE_HOST_DEVICE inline float SobelMagnitudeAt(const float* in, int width,
                                               int height, int x, int y) {
  const int xm = ClampIndex(x - 1, width);
  const int xp = ClampIndex(x + 1, width);
  const int64_t ym = static_cast<int64_t>(ClampIndex(y - 1, height)) * width;
  const int64_t y0 = static_cast<int64_t>(y) * width;
  const int64_t yp = static_cast<int64_t>(ClampIndex(y + 1, height)) * width;

  const float gx = (in[ym + xp] + 2.0f * in[y0 + xp] + in[yp + xp]) -
                   (in[ym + xm] + 2.0f * in[y0 + xm] + in[yp + xm]);
  const float gy = (in[yp + xm] + 2.0f * in[yp + x] + in[yp + xp]) -
                   (in[ym + xm] + 2.0f * in[ym + x] + in[ym + xp]);
  return sqrtf(gx * gx + gy * gy);
}

// Converts a gradient magnitude to an 8-bit output pixel. With a threshold
// of 0 the (clamped) magnitude is written; otherwise a binary edge map.
EDGE_HOST_DEVICE inline uint8_t EdgeToByte(float magnitude, float threshold) {
  if (threshold > 0.0f) {
    return magnitude >= threshold ? 255 : 0;
  }
  const float clamped = magnitude > 255.0f ? 255.0f : magnitude;
  return static_cast<uint8_t>(clamped + 0.5f);
}

// Converts a float intensity to an 8-bit pixel (used for the blurred image).
EDGE_HOST_DEVICE inline uint8_t FloatToByte(float value) {
  const float clamped = value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);
  return static_cast<uint8_t>(clamped + 0.5f);
}

}  // namespace edge

#endif  // SRC_PIXEL_OPS_H_
