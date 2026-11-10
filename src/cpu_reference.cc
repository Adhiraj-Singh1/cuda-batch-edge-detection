// Copyright 2026 Adhiraj Singh

#include "src/cpu_reference.h"

#include <math.h>

#include <vector>

#include "src/pixel_ops.h"

namespace edge {

int BuildGaussianWeights(float sigma, std::vector<float>* weights) {
  int radius = static_cast<int>(ceilf(3.0f * sigma));
  if (radius < 1) radius = 1;
  if (radius > kMaxGaussianRadius) radius = kMaxGaussianRadius;

  weights->assign(2 * radius + 1, 0.0f);
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const float w = expf(-(k * k) / (2.0f * sigma * sigma));
    (*weights)[k + radius] = w;
    sum += w;
  }
  for (float& w : *weights) w /= sum;
  return radius;
}

void RunCpuPipeline(const uint8_t* rgb, int width, int height,
                    const std::vector<float>& weights, int radius,
                    float threshold, std::vector<uint8_t>* edges) {
  const size_t num_pixels = static_cast<size_t>(width) * height;
  std::vector<float> gray(num_pixels);
  std::vector<float> temp(num_pixels);
  std::vector<float> blurred(num_pixels);
  edges->resize(num_pixels);

  for (size_t i = 0; i < num_pixels; ++i) {
    gray[i] = GrayFromRgb(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      temp[static_cast<size_t>(y) * width + x] =
          GaussianRowAt(gray.data(), width, x, y, weights.data(), radius);
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      blurred[static_cast<size_t>(y) * width + x] = GaussianColumnAt(
          temp.data(), width, height, x, y, weights.data(), radius);
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float magnitude =
          SobelMagnitudeAt(blurred.data(), width, height, x, y);
      (*edges)[static_cast<size_t>(y) * width + x] =
          EdgeToByte(magnitude, threshold);
    }
  }
}

}  // namespace edge
