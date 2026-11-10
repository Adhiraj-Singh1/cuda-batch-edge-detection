// Copyright 2026 Adhiraj Singh
//
// generate_dataset: writes a reproducible set of synthetic RGB test images
// (gradient backgrounds, random filled/outlined shapes, lines and sensor-like
// Gaussian noise) as PNG files. The same seed always produces the same data.
//
// Usage: generate_dataset -o DIR -c COUNT -W WIDTH -H HEIGHT
//                         [-p PREFIX] [-s SEED] [-n NOISE_SIGMA]

#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"

namespace {

struct Color {
  float r, g, b;
};

class Canvas {
 public:
  Canvas(int width, int height)
      : width_(width), height_(height), pixels_(width * height * 3, 0.0f) {}

  void FillGradient(Color a, Color b, float angle) {
    const float dx = cosf(angle);
    const float dy = sinf(angle);
    const float span = fabsf(dx) * width_ + fabsf(dy) * height_;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        float t = (x * dx + y * dy) / span;
        t = t - floorf(t);
        Set(x, y, {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t,
                   a.b + (b.b - a.b) * t});
      }
    }
  }

  void FillCircle(float cx, float cy, float radius, Color c, bool ring) {
    const float inner = ring ? radius * 0.7f : -1.0f;
    for (int y = Lo(cy - radius); y <= Hi(cy + radius, height_); ++y) {
      for (int x = Lo(cx - radius); x <= Hi(cx + radius, width_); ++x) {
        const float d = hypotf(x - cx, y - cy);
        if (d <= radius && d >= inner) Set(x, y, c);
      }
    }
  }

  void FillRectangle(float x0, float y0, float x1, float y1, Color c) {
    for (int y = Lo(y0); y <= Hi(y1, height_); ++y) {
      for (int x = Lo(x0); x <= Hi(x1, width_); ++x) Set(x, y, c);
    }
  }

  void FillTriangle(float ax, float ay, float bx, float by, float cx,
                    float cy, Color c) {
    const int min_x = Lo(std::min({ax, bx, cx}));
    const int max_x = Hi(std::max({ax, bx, cx}), width_);
    const int min_y = Lo(std::min({ay, by, cy}));
    const int max_y = Hi(std::max({ay, by, cy}), height_);
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        const float w0 = Edge(bx, by, cx, cy, x, y);
        const float w1 = Edge(cx, cy, ax, ay, x, y);
        const float w2 = Edge(ax, ay, bx, by, x, y);
        if ((w0 >= 0 && w1 >= 0 && w2 >= 0) ||
            (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
          Set(x, y, c);
        }
      }
    }
  }

  void DrawLine(float x0, float y0, float x1, float y1, float thickness,
                Color c) {
    const float length = std::max(1.0f, hypotf(x1 - x0, y1 - y0));
    const int steps = static_cast<int>(length * 2);
    for (int i = 0; i <= steps; ++i) {
      const float t = static_cast<float>(i) / steps;
      FillCircle(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, thickness, c, false);
    }
  }

  void AddNoise(float sigma, std::mt19937* rng) {
    if (sigma <= 0.0f) return;
    std::normal_distribution<float> noise(0.0f, sigma);
    for (float& v : pixels_) v += noise(*rng);
  }

  std::vector<uint8_t> ToBytes() const {
    std::vector<uint8_t> out(pixels_.size());
    for (size_t i = 0; i < pixels_.size(); ++i) {
      const float v = std::min(255.0f, std::max(0.0f, pixels_[i]));
      out[i] = static_cast<uint8_t>(v + 0.5f);
    }
    return out;
  }

 private:
  static float Edge(float ax, float ay, float bx, float by, float px,
                    float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
  }
  static int Lo(float v) { return std::max(0, static_cast<int>(floorf(v))); }
  static int Hi(float v, int size) {
    return std::min(size - 1, static_cast<int>(ceilf(v)));
  }
  void Set(int x, int y, Color c) {
    if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
    float* p = &pixels_[(static_cast<size_t>(y) * width_ + x) * 3];
    p[0] = c.r;
    p[1] = c.g;
    p[2] = c.b;
  }

  int width_;
  int height_;
  std::vector<float> pixels_;
};

void DrawScene(Canvas* canvas, int width, int height, std::mt19937* rng) {
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  auto random_color = [&]() {
    return Color{unit(*rng) * 255.0f, unit(*rng) * 255.0f,
                 unit(*rng) * 255.0f};
  };
  const float scale = static_cast<float>(std::min(width, height));

  canvas->FillGradient(random_color(), random_color(),
                       unit(*rng) * 6.2831853f);
  const int num_shapes = 6 + static_cast<int>(unit(*rng) * 10);
  for (int s = 0; s < num_shapes; ++s) {
    const float cx = unit(*rng) * width;
    const float cy = unit(*rng) * height;
    const float size = scale * (0.04f + unit(*rng) * 0.18f);
    const Color c = random_color();
    switch (static_cast<int>(unit(*rng) * 5)) {
      case 0:
        canvas->FillCircle(cx, cy, size, c, false);
        break;
      case 1:
        canvas->FillCircle(cx, cy, size, c, true);
        break;
      case 2:
        canvas->FillRectangle(cx - size, cy - size * 0.6f, cx + size,
                              cy + size * 0.6f, c);
        break;
      case 3:
        canvas->FillTriangle(cx, cy - size, cx - size, cy + size, cx + size,
                             cy + size * unit(*rng), c);
        break;
      default:
        canvas->DrawLine(cx, cy, unit(*rng) * width, unit(*rng) * height,
                         1.0f + scale * 0.006f * unit(*rng), c);
        break;
    }
  }
}

// mkdir -p: creates every missing directory along `path`.
void MakeDirectories(const std::string& path) {
  for (size_t pos = path.find('/', 1); pos != std::string::npos;
       pos = path.find('/', pos + 1)) {
    mkdir(path.substr(0, pos).c_str(), 0755);
  }
  mkdir(path.c_str(), 0755);
}

void PrintUsage(const char* name) {
  fprintf(stderr,
          "Usage: %s -o DIR -c COUNT -W WIDTH -H HEIGHT [-p PREFIX] "
          "[-s SEED] [-n NOISE_SIGMA]\n",
          name);
}

}  // namespace

int main(int argc, char** argv) {
  std::string output_dir = "data/input";
  std::string prefix = "image";
  int count = 100;
  int width = 256;
  int height = 256;
  unsigned int seed = 2026;
  float noise_sigma = 6.0f;

  int opt = 0;
  while ((opt = getopt(argc, argv, "o:c:W:H:p:s:n:h")) != -1) {
    switch (opt) {
      case 'o': output_dir = optarg; break;
      case 'c': count = atoi(optarg); break;
      case 'W': width = atoi(optarg); break;
      case 'H': height = atoi(optarg); break;
      case 'p': prefix = optarg; break;
      case 's': seed = static_cast<unsigned int>(strtoul(optarg, nullptr, 10));
                break;
      case 'n': noise_sigma = static_cast<float>(atof(optarg)); break;
      default:
        PrintUsage(argv[0]);
        return opt == 'h' ? 0 : 2;
    }
  }
  if (count <= 0 || width <= 0 || height <= 0) {
    PrintUsage(argv[0]);
    return 2;
  }
  MakeDirectories(output_dir);

  std::mt19937 rng(seed);
  for (int i = 0; i < count; ++i) {
    Canvas canvas(width, height);
    DrawScene(&canvas, width, height, &rng);
    canvas.AddNoise(noise_sigma, &rng);
    const std::vector<uint8_t> bytes = canvas.ToBytes();
    char name[512];
    snprintf(name, sizeof(name), "%s/%s_%04d.png", output_dir.c_str(),
             prefix.c_str(), i);
    if (!stbi_write_png(name, width, height, 3, bytes.data(), width * 3)) {
      fprintf(stderr, "Failed to write %s\n", name);
      return 1;
    }
  }
  printf("Wrote %d images of %dx%d to %s (prefix '%s', seed %u)\n", count,
         width, height, output_dir.c_str(), prefix.c_str(), seed);
  return 0;
}
