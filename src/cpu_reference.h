// Copyright 2026 Adhiraj Singh
//
// Single-threaded CPU implementation of the same pipeline, used to verify the
// GPU output and to measure the GPU speedup.

#ifndef SRC_CPU_REFERENCE_H_
#define SRC_CPU_REFERENCE_H_

#include <stdint.h>

#include <vector>

namespace edge {

// Fills `weights` (size 2 * radius + 1) with a normalized 1-D Gaussian and
// returns the radius used for the given sigma.
int BuildGaussianWeights(float sigma, std::vector<float>* weights);

// Runs grayscale -> separable Gaussian blur -> Sobel on an RGB image.
// `edges` receives width * height bytes.
void RunCpuPipeline(const uint8_t* rgb, int width, int height,
                    const std::vector<float>& weights, int radius,
                    float threshold, std::vector<uint8_t>* edges);

}  // namespace edge

#endif  // SRC_CPU_REFERENCE_H_
