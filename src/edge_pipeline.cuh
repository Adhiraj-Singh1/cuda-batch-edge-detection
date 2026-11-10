// Copyright 2026 Adhiraj Singh
//
// GPU edge-detection pipeline: RGB -> grayscale -> separable Gaussian blur ->
// Sobel gradient magnitude. Each StreamWorker owns one CUDA stream plus its
// own pinned host and device buffers, so several images can be in flight at
// once (copies of one image overlap with kernels of another).

#ifndef SRC_EDGE_PIPELINE_CUH_
#define SRC_EDGE_PIPELINE_CUH_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t check_cuda_status = (call);                                \
    if (check_cuda_status != cudaSuccess) {                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
              cudaGetErrorString(check_cuda_status), __FILE__, __LINE__);  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

namespace edge {

// Copies the Gaussian weights into constant memory (shared by all streams).
void UploadGaussianWeights(const std::vector<float>& weights);

class StreamWorker {
 public:
  explicit StreamWorker(int id);
  ~StreamWorker();

  StreamWorker(const StreamWorker&) = delete;
  StreamWorker& operator=(const StreamWorker&) = delete;

  // Grows the host/device buffers so an image of `num_pixels` fits.
  void EnsureCapacity(size_t num_pixels);

  // Pinned host buffer the caller fills with RGB data before Enqueue().
  uint8_t* host_rgb() { return host_rgb_; }
  size_t rgb_capacity_bytes() const { return capacity_pixels_ * 3; }

  // Queues H2D copy, the four kernels and the D2H copy on this stream.
  // Returns immediately; call Wait() before reading the results.
  void Enqueue(int width, int height, int radius, float threshold,
               dim3 block, bool copy_blurred);

  // Blocks until this stream is idle and returns the GPU time (ms) of the
  // last enqueued image, measured with CUDA events.
  float Wait();

  const uint8_t* host_edges() const { return host_edges_; }
  const uint8_t* host_blurred() const { return host_blurred_; }
  int id() const { return id_; }

 private:
  void FreeBuffers();

  int id_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  size_t capacity_pixels_ = 0;

  // Pinned (page-locked) host memory, required for truly async copies.
  uint8_t* host_rgb_ = nullptr;
  uint8_t* host_edges_ = nullptr;
  uint8_t* host_blurred_ = nullptr;

  // Device memory.
  uint8_t* dev_rgb_ = nullptr;
  float* dev_gray_ = nullptr;
  float* dev_temp_ = nullptr;
  float* dev_blurred_ = nullptr;
  uint8_t* dev_edges_ = nullptr;
  uint8_t* dev_blurred_bytes_ = nullptr;
};

}  // namespace edge

#endif  // SRC_EDGE_PIPELINE_CUH_
