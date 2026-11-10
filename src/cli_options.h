// Copyright 2026 Adhiraj Singh
//
// Command line options for the edge_detect executable.

#ifndef SRC_CLI_OPTIONS_H_
#define SRC_CLI_OPTIONS_H_

#include <string>

namespace edge {

struct Options {
  std::string input_dir = "data/input";
  std::string output_dir = "output";
  std::string log_file = "logs/run_log.txt";
  std::string summary_file = "logs/summary.md";
  float sigma = 1.4f;       // Gaussian standard deviation in pixels.
  float threshold = 0.0f;   // 0 = write gradient magnitude, >0 = binary map.
  int num_streams = 4;      // Number of concurrent CUDA streams.
  int block_size = 16;      // Threads per block = block_size x block_size.
  int max_images = 0;       // 0 = process every image in input_dir.
  int gallery_count = 6;    // Side-by-side previews written to the gallery.
  int device = -1;          // -1 = pick the GPU with the most SMs.
  bool verify = false;      // Compare every GPU result with the CPU result.
};

// Parses argv into `options`. On failure returns false and sets `error`.
// Sets `show_help` when -h/--help was given.
bool ParseOptions(int argc, char** argv, Options* options, bool* show_help,
                  std::string* error);

void PrintUsage(const char* program_name);

}  // namespace edge

#endif  // SRC_CLI_OPTIONS_H_
