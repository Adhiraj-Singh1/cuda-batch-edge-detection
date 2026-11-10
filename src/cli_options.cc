// Copyright 2026 Adhiraj Singh

#include "src/cli_options.h"

#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

namespace edge {
namespace {

bool ParseInt(const char* text, int* value) {
  char* end = nullptr;
  const int64_t parsed = strtoll(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseFloat(const char* text, float* value) {
  char* end = nullptr;
  const float parsed = strtof(text, &end);
  if (end == text || *end != '\0') return false;
  *value = parsed;
  return true;
}

}  // namespace

bool ParseOptions(int argc, char** argv, Options* options, bool* show_help,
                  std::string* error) {
  static const struct option kLongOptions[] = {
      {"input_dir", required_argument, nullptr, 'i'},
      {"output_dir", required_argument, nullptr, 'o'},
      {"log_file", required_argument, nullptr, 'l'},
      {"summary_file", required_argument, nullptr, 'S'},
      {"sigma", required_argument, nullptr, 's'},
      {"threshold", required_argument, nullptr, 't'},
      {"num_streams", required_argument, nullptr, 'n'},
      {"block_size", required_argument, nullptr, 'b'},
      {"max_images", required_argument, nullptr, 'm'},
      {"gallery", required_argument, nullptr, 'g'},
      {"device", required_argument, nullptr, 'd'},
      {"verify", no_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  *show_help = false;
  optind = 1;
  int opt = 0;
  while ((opt = getopt_long(argc, argv, "i:o:l:S:s:t:n:b:m:g:d:vh",
                            kLongOptions, nullptr)) != -1) {
    bool ok = true;
    switch (opt) {
      case 'i': options->input_dir = optarg; break;
      case 'o': options->output_dir = optarg; break;
      case 'l': options->log_file = optarg; break;
      case 'S': options->summary_file = optarg; break;
      case 's': ok = ParseFloat(optarg, &options->sigma); break;
      case 't': ok = ParseFloat(optarg, &options->threshold); break;
      case 'n': ok = ParseInt(optarg, &options->num_streams); break;
      case 'b': ok = ParseInt(optarg, &options->block_size); break;
      case 'm': ok = ParseInt(optarg, &options->max_images); break;
      case 'g': ok = ParseInt(optarg, &options->gallery_count); break;
      case 'd': ok = ParseInt(optarg, &options->device); break;
      case 'v': options->verify = true; break;
      case 'h': *show_help = true; return true;
      default:
        *error = "unknown option";
        return false;
    }
    if (!ok) {
      *error = std::string("invalid value '") + optarg + "' for -" +
               static_cast<char>(opt);
      return false;
    }
  }

  if (options->sigma < 0.1f || options->sigma > 5.0f) {
    *error = "--sigma must be between 0.1 and 5.0";
  } else if (options->threshold < 0.0f) {
    *error = "--threshold must be >= 0";
  } else if (options->num_streams < 1 || options->num_streams > 32) {
    *error = "--num_streams must be between 1 and 32";
  } else if (options->block_size < 1 ||
             options->block_size * options->block_size > 1024) {
    *error = "--block_size must be >= 1 and block_size^2 <= 1024";
  } else if (options->max_images < 0 || options->gallery_count < 0) {
    *error = "--max_images and --gallery must be >= 0";
  } else {
    return true;
  }
  return false;
}

void PrintUsage(const char* program_name) {
  printf(
      "Usage: %s [options]\n"
      "Batch GPU edge detection: grayscale -> Gaussian blur -> Sobel.\n\n"
      "  -i, --input_dir DIR     folder of input images (default data/input)\n"
      "  -o, --output_dir DIR    output folder (default output)\n"
      "  -l, --log_file FILE     run log (default logs/run_log.txt)\n"
      "  -S, --summary_file FILE markdown summary (default logs/summary.md)\n"
      "  -s, --sigma F           Gaussian sigma in pixels (default 1.4)\n"
      "  -t, --threshold F       0 = gradient magnitude, >0 = binary edges\n"
      "  -n, --num_streams N     concurrent CUDA streams (default 4)\n"
      "  -b, --block_size N      threads per block = N x N (default 16)\n"
      "  -m, --max_images N      process at most N images (default all)\n"
      "  -g, --gallery N         side-by-side previews to write (default 6)\n"
      "  -d, --device N          GPU index (default: GPU with most SMs)\n"
      "  -v, --verify            check every GPU result against the CPU\n"
      "  -h, --help              show this help\n",
      program_name);
}

}  // namespace edge
