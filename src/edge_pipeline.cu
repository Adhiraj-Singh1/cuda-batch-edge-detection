// Copyright 2026 Adhiraj Singh

#include "src/edge_pipeline.cuh"

#include <vector>

#include "src/pixel_ops.h"

namespace edge {
namespace {

// Gaussian weights live in constant memory: every thread of a warp reads the
// same weight at the same time, which the constant cache broadcasts.
__constant__ float c_gaussian_weights[kMaxGaussianTaps];

__global__ void RgbToGrayKernel(const uint8_t* __restrict__ rgb,
                                float* __restrict__ gray, int width,
                                int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int64_t i = static_cast<int64_t>(y) * width + x;
  gray[i] = GrayFromRgb(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]);
}

__global__ void GaussianRowKernel(const float* __restrict__ in,
                                  float* __restrict__ out, int width,
                                  int height, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  out[static_cast<int64_t>(y) * width + x] =
      GaussianRowAt(in, width, x, y, c_gaussian_weights, radius);
}

__global__ void GaussianColumnKernel(const float* __restrict__ in,
                                     float* __restrict__ out, int width,
                                     int height, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  out[static_cast<int64_t>(y) * width + x] =
      GaussianColumnAt(in, width, height, x, y, c_gaussian_weights, radius);
}

__global__ void SobelKernel(const float* __restrict__ blurred,
                            uint8_t* __restrict__ edges, int width,
                            int height, float threshold) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  edges[static_cast<int64_t>(y) * width + x] =
      EdgeToByte(SobelMagnitudeAt(blurred, width, height, x, y), threshold);
}

__global__ void FloatToByteKernel(const float* __restrict__ in,
                                  uint8_t* __restrict__ out, int width,
                                  int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int64_t i = static_cast<int64_t>(y) * width + x;
  out[i] = FloatToByte(in[i]);
}

}  // namespace

void UploadGaussianWeights(const std::vector<float>& weights) {
  CHECK_CUDA(cudaMemcpyToSymbol(c_gaussian_weights, weights.data(),
                                weights.size() * sizeof(float)));
}

StreamWorker::StreamWorker(int id) : id_(id) {
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  CHECK_CUDA(cudaEventCreate(&start_));
  CHECK_CUDA(cudaEventCreate(&stop_));
}

StreamWorker::~StreamWorker() {
  cudaStreamSynchronize(stream_);
  FreeBuffers();
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
  cudaStreamDestroy(stream_);
}

void StreamWorker::FreeBuffers() {
  cudaFreeHost(host_rgb_);
  cudaFreeHost(host_edges_);
  cudaFreeHost(host_blurred_);
  cudaFree(dev_rgb_);
  cudaFree(dev_gray_);
  cudaFree(dev_temp_);
  cudaFree(dev_blurred_);
  cudaFree(dev_edges_);
  cudaFree(dev_blurred_bytes_);
  host_rgb_ = host_edges_ = host_blurred_ = nullptr;
  dev_rgb_ = dev_edges_ = dev_blurred_bytes_ = nullptr;
  dev_gray_ = dev_temp_ = dev_blurred_ = nullptr;
  capacity_pixels_ = 0;
}

void StreamWorker::EnsureCapacity(size_t num_pixels) {
  if (num_pixels <= capacity_pixels_) return;
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  FreeBuffers();
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_rgb_),
                           num_pixels * 3, cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_edges_), num_pixels,
                           cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_blurred_),
                           num_pixels, cudaHostAllocDefault));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_rgb_), num_pixels * 3));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_gray_),
                        num_pixels * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_temp_),
                        num_pixels * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_blurred_),
                        num_pixels * sizeof(float)));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_edges_), num_pixels));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_blurred_bytes_),
                        num_pixels));
  capacity_pixels_ = num_pixels;
}

void StreamWorker::Enqueue(int width, int height, int radius, float threshold,
                           dim3 block, bool copy_blurred) {
  const size_t num_pixels = static_cast<size_t>(width) * height;
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

  CHECK_CUDA(cudaEventRecord(start_, stream_));
  CHECK_CUDA(cudaMemcpyAsync(dev_rgb_, host_rgb_, num_pixels * 3,
                             cudaMemcpyHostToDevice, stream_));

  RgbToGrayKernel<<<grid, block, 0, stream_>>>(dev_rgb_, dev_gray_, width,
                                               height);
  GaussianRowKernel<<<grid, block, 0, stream_>>>(dev_gray_, dev_temp_, width,
                                                 height, radius);
  GaussianColumnKernel<<<grid, block, 0, stream_>>>(dev_temp_, dev_blurred_,
                                                    width, height, radius);
  SobelKernel<<<grid, block, 0, stream_>>>(dev_blurred_, dev_edges_, width,
                                           height, threshold);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpyAsync(host_edges_, dev_edges_, num_pixels,
                             cudaMemcpyDeviceToHost, stream_));
  if (copy_blurred) {
    FloatToByteKernel<<<grid, block, 0, stream_>>>(
        dev_blurred_, dev_blurred_bytes_, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(host_blurred_, dev_blurred_bytes_, num_pixels,
                               cudaMemcpyDeviceToHost, stream_));
  }
  CHECK_CUDA(cudaEventRecord(stop_, stream_));
}

float StreamWorker::Wait() {
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
  return elapsed_ms;
}

}  // namespace edge
