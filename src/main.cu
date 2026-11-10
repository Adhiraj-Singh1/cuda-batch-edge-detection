// Copyright 2026 Adhiraj Singh
//
// edge_detect: runs a GPU edge-detection pipeline over every image in a
// folder using several CUDA streams, writes the edge maps, a gallery of
// side-by-side previews, a per-image log and a markdown summary.

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "src/cli_options.h"
#include "src/cpu_reference.h"
#include "src/edge_pipeline.cuh"
#include "src/image_io.h"
#include "src/pixel_ops.h"

namespace edge {
namespace {

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

// Writes every message both to stdout and to the log file.
class RunLogger {
 public:
  explicit RunLogger(const std::string& path)
      : file_(fopen(path.c_str(), "w")) {}
  ~RunLogger() {
    if (file_ != nullptr) fclose(file_);
  }
  RunLogger(const RunLogger&) = delete;
  RunLogger& operator=(const RunLogger&) = delete;

  bool ok() const { return file_ != nullptr; }

  __attribute__((format(printf, 2, 3))) void Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    vprintf(format, args);
    if (file_ != nullptr) {
      vfprintf(file_, format, args_copy);
      fflush(file_);
    }
    va_end(args_copy);
    va_end(args);
  }

 private:
  FILE* file_;
};

// An image that has been enqueued on a stream and not yet written out.
struct PendingImage {
  bool active = false;
  int index = 0;
  std::string path;
  int width = 0;
  int height = 0;
  bool in_gallery = false;
};

struct BatchStats {
  int processed = 0;
  int failed = 0;
  int verified_ok = 0;
  int verified_bad = 0;
  int max_difference = 0;
  int64_t total_pixels = 0;
  double gpu_ms = 0.0;
  double cpu_ms = 0.0;
};

int SelectDevice(int requested, RunLogger* log) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    log->Log("ERROR: no CUDA capable GPU was found.\n");
    exit(EXIT_FAILURE);
  }
  int chosen = requested;
  if (chosen < 0 || chosen >= device_count) {
    int best_sms = -1;
    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp prop;
      CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
      if (prop.multiProcessorCount > best_sms) {
        best_sms = prop.multiProcessorCount;
        chosen = i;
      }
    }
  }
  CHECK_CUDA(cudaSetDevice(chosen));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, chosen));
  log->Log("GPU devices found: %d\n", device_count);
  log->Log("Using GPU %d: %s | compute %d.%d | %d SMs | %.1f GB global mem\n",
           chosen, prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  return chosen;
}

// Builds a 3-panel preview: grayscale input | blurred | edges.
void WriteGalleryImage(const std::string& path, const uint8_t* rgb,
                       const uint8_t* blurred, const uint8_t* edges,
                       int width, int height) {
  const int panel_width = width * 3;
  std::vector<uint8_t> panel(static_cast<size_t>(panel_width) * height);
  for (int y = 0; y < height; ++y) {
    uint8_t* row = panel.data() + static_cast<size_t>(y) * panel_width;
    for (int x = 0; x < width; ++x) {
      const size_t i = static_cast<size_t>(y) * width + x;
      row[x] = FloatToByte(
          GrayFromRgb(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]));
      row[width + x] = blurred[i];
      row[2 * width + x] = edges[i];
    }
  }
  WriteGrayPng(path, panel.data(), panel_width, height);
}

// Waits for `worker` to finish `pending`, then writes and (optionally)
// verifies its results.
void FinishImage(StreamWorker* worker, PendingImage* pending,
                 const Options& options, const std::vector<float>& weights,
                 int radius, int total_images, BatchStats* stats,
                 RunLogger* log) {
  if (!pending->active) return;
  const float gpu_ms = worker->Wait();
  const std::string base = BaseNameWithoutExtension(pending->path);
  const std::string edge_path =
      options.output_dir + "/edges/" + base + "_edges.png";
  const int width = pending->width;
  const int height = pending->height;

  const bool written =
      WriteGrayPng(edge_path, worker->host_edges(), width, height);

  std::string verify_text;
  if (options.verify) {
    std::vector<uint8_t> cpu_edges;
    const Clock::time_point cpu_start = Clock::now();
    RunCpuPipeline(worker->host_rgb(), width, height, weights, radius,
                   options.threshold, &cpu_edges);
    const double cpu_ms = MillisecondsSince(cpu_start);
    stats->cpu_ms += cpu_ms;

    int max_diff = 0;
    const size_t num_pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < num_pixels; ++i) {
      const int diff = abs(static_cast<int>(cpu_edges[i]) -
                           static_cast<int>(worker->host_edges()[i]));
      max_diff = std::max(max_diff, diff);
    }
    // Allow +/-1 for float rounding differences (e.g. fused multiply-add on
    // the GPU) when writing the gradient magnitude.
    const int tolerance = options.threshold > 0.0f ? 0 : 1;
    const bool match = max_diff <= tolerance;
    stats->max_difference = std::max(stats->max_difference, max_diff);
    if (match) {
      ++stats->verified_ok;
    } else {
      ++stats->verified_bad;
    }
    char buffer[96];
    snprintf(buffer, sizeof(buffer), " | cpu %8.3f ms | max diff %d %s",
             cpu_ms, max_diff, match ? "OK" : "MISMATCH");
    verify_text = buffer;
  }

  if (pending->in_gallery) {
    WriteGalleryImage(options.output_dir + "/gallery/" + base + "_gallery.png",
                      worker->host_rgb(), worker->host_blurred(),
                      worker->host_edges(), width, height);
  }

  stats->gpu_ms += gpu_ms;
  stats->total_pixels += static_cast<int64_t>(width) * height;
  if (written) {
    ++stats->processed;
  } else {
    ++stats->failed;
  }
  log->Log("[%4d/%4d] stream %d | %-28s %5dx%-5d | gpu %7.3f ms%s -> %s%s\n",
           pending->index + 1, total_images, worker->id(),
           (base + ".png").c_str(), width, height, gpu_ms,
           verify_text.c_str(), edge_path.c_str(),
           written ? "" : " (WRITE FAILED)");
  pending->active = false;
}

void WriteSummary(const Options& options, const BatchStats& stats,
                  int total_images, double wall_ms, int radius,
                  RunLogger* log) {
  const double megapixels = stats.total_pixels / 1.0e6;
  log->Log("\n==================== Summary ====================\n");
  log->Log("Images found:                 %d\n", total_images);
  log->Log("Images processed:             %d\n", stats.processed);
  log->Log("Images failed:                %d\n", stats.failed);
  log->Log("Total pixels:                 %.2f MP\n", megapixels);
  log->Log("Gaussian sigma / radius:      %.2f / %d (%d taps)\n",
           options.sigma, radius, 2 * radius + 1);
  log->Log("CUDA streams / block:         %d / %dx%d\n", options.num_streams,
           options.block_size, options.block_size);
  log->Log("Wall time (incl. file I/O):   %.1f ms\n", wall_ms);
  log->Log("Sum of per-image GPU time:    %.1f ms\n", stats.gpu_ms);
  log->Log("Throughput (wall clock):      %.1f MP/s\n",
           megapixels / (wall_ms / 1000.0));
  if (options.verify) {
    log->Log("Sum of CPU reference time:    %.1f ms\n", stats.cpu_ms);
    log->Log("GPU speedup (compute+copies): %.1fx\n",
             stats.gpu_ms > 0 ? stats.cpu_ms / stats.gpu_ms : 0.0);
    log->Log("Verification:                 %d OK, %d mismatched "
             "(max pixel difference %d)\n",
             stats.verified_ok, stats.verified_bad, stats.max_difference);
  }

  FILE* md = fopen(options.summary_file.c_str(), "w");
  if (md == nullptr) return;
  fprintf(md, "# Run summary\n\n| Metric | Value |\n|---|---|\n");
  fprintf(md, "| Images processed | %d of %d |\n", stats.processed,
          total_images);
  fprintf(md, "| Total pixels | %.2f MP |\n", megapixels);
  fprintf(md, "| Gaussian sigma / radius | %.2f / %d |\n", options.sigma,
          radius);
  fprintf(md, "| CUDA streams / block size | %d / %dx%d |\n",
          options.num_streams, options.block_size, options.block_size);
  fprintf(md, "| Wall time (incl. file I/O) | %.1f ms |\n", wall_ms);
  fprintf(md, "| Sum of per-image GPU time | %.1f ms |\n", stats.gpu_ms);
  fprintf(md, "| Throughput (wall clock) | %.1f MP/s |\n",
          megapixels / (wall_ms / 1000.0));
  if (options.verify) {
    fprintf(md, "| Sum of CPU reference time | %.1f ms |\n", stats.cpu_ms);
    fprintf(md, "| GPU speedup vs. 1 CPU core | %.1fx |\n",
            stats.gpu_ms > 0 ? stats.cpu_ms / stats.gpu_ms : 0.0);
    fprintf(md, "| Verified against CPU | %d OK, %d mismatched |\n",
            stats.verified_ok, stats.verified_bad);
  }
  fclose(md);
}

int Run(const Options& options) {
  const size_t slash = options.log_file.find_last_of('/');
  if (slash != std::string::npos) {
    MakeDirectories(options.log_file.substr(0, slash));
  }
  RunLogger log(options.log_file);
  if (!log.ok()) {
    fprintf(stderr, "Cannot open log file %s\n", options.log_file.c_str());
    return EXIT_FAILURE;
  }

  const time_t now = time(nullptr);
  log.Log("CUDA batch edge detection - run started %s", ctime(&now));
  SelectDevice(options.device, &log);

  std::vector<std::string> files = ListImageFiles(options.input_dir);
  if (options.max_images > 0 &&
      static_cast<int>(files.size()) > options.max_images) {
    files.resize(options.max_images);
  }
  if (files.empty()) {
    log.Log("ERROR: no images found in %s\n", options.input_dir.c_str());
    return EXIT_FAILURE;
  }
  const int total = static_cast<int>(files.size());
  log.Log("Input folder: %s (%d images)\n", options.input_dir.c_str(), total);

  if (!MakeDirectories(options.output_dir + "/edges") ||
      !MakeDirectories(options.output_dir + "/gallery")) {
    log.Log("ERROR: cannot create output folders in %s\n",
            options.output_dir.c_str());
    return EXIT_FAILURE;
  }

  std::vector<float> weights;
  const int radius = BuildGaussianWeights(options.sigma, &weights);
  UploadGaussianWeights(weights);

  const int num_streams = std::min(options.num_streams, total);
  std::vector<std::unique_ptr<StreamWorker>> workers;
  std::vector<PendingImage> pending(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    workers.emplace_back(new StreamWorker(i));
  }
  const int gallery_step =
      options.gallery_count > 0 ? std::max(1, total / options.gallery_count)
                                : 0;
  const dim3 block(options.block_size, options.block_size);
  log.Log("Processing with %d CUDA streams, %dx%d threads per block\n\n",
          num_streams, options.block_size, options.block_size);

  BatchStats stats;
  int gallery_written = 0;
  const Clock::time_point wall_start = Clock::now();
  for (int i = 0; i < total; ++i) {
    const int slot = i % num_streams;
    StreamWorker* worker = workers[slot].get();
    // Finish whatever this stream was doing before reusing its buffers.
    // Meanwhile the other streams keep the GPU busy.
    FinishImage(worker, &pending[slot], options, weights, radius, total,
                &stats, &log);

    int width = 0;
    int height = 0;
    if (!ProbeImage(files[i], &width, &height)) {
      log.Log("[%4d/%4d] SKIPPED unreadable image %s\n", i + 1, total,
              files[i].c_str());
      ++stats.failed;
      continue;
    }
    worker->EnsureCapacity(static_cast<size_t>(width) * height);
    if (!LoadRgbImage(files[i], worker->host_rgb(),
                      worker->rgb_capacity_bytes(), &width, &height)) {
      log.Log("[%4d/%4d] SKIPPED cannot decode %s\n", i + 1, total,
              files[i].c_str());
      ++stats.failed;
      continue;
    }

    PendingImage& job = pending[slot];
    job.active = true;
    job.index = i;
    job.path = files[i];
    job.width = width;
    job.height = height;
    job.in_gallery = gallery_step > 0 && i % gallery_step == 0 &&
                     gallery_written < options.gallery_count;
    if (job.in_gallery) ++gallery_written;
    worker->Enqueue(width, height, radius, options.threshold, block,
                    job.in_gallery);
  }
  for (int slot = 0; slot < num_streams; ++slot) {
    FinishImage(workers[slot].get(), &pending[slot], options, weights, radius,
                total, &stats, &log);
  }
  const double wall_ms = MillisecondsSince(wall_start);

  WriteSummary(options, stats, total, wall_ms, radius, &log);
  workers.clear();
  CHECK_CUDA(cudaDeviceReset());
  return stats.failed == 0 && stats.verified_bad == 0 ? EXIT_SUCCESS
                                                      : EXIT_FAILURE;
}

}  // namespace
}  // namespace edge

int main(int argc, char** argv) {
  edge::Options options;
  bool show_help = false;
  std::string error;
  if (!edge::ParseOptions(argc, argv, &options, &show_help, &error)) {
    fprintf(stderr, "Error: %s\n\n", error.c_str());
    edge::PrintUsage(argv[0]);
    return 2;
  }
  if (show_help) {
    edge::PrintUsage(argv[0]);
    return 0;
  }
  return edge::Run(options);
}
