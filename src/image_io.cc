// Copyright 2026 Adhiraj Singh

#include "src/image_io.h"

#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

// stb is third-party code; silence its warnings only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "third_party/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#pragma GCC diagnostic pop

namespace edge {
namespace {

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

bool HasSupportedExtension(const std::string& name) {
  static const char* const kExtensions[] = {".png", ".jpg", ".jpeg", ".bmp",
                                            ".tga", ".pgm", ".ppm"};
  const std::string lower = ToLower(name);
  for (const char* ext : kExtensions) {
    const size_t len = strlen(ext);
    if (lower.size() > len &&
        lower.compare(lower.size() - len, len, ext) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::vector<std::string> ListImageFiles(const std::string& directory) {
  std::vector<std::string> files;
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) return files;
  while (struct dirent* entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name.empty() || name[0] == '.') continue;
    if (!HasSupportedExtension(name)) continue;
    files.push_back(directory + "/" + name);
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

bool ProbeImage(const std::string& path, int* width, int* height) {
  int channels = 0;
  return stbi_info(path.c_str(), width, height, &channels) == 1;
}

bool LoadRgbImage(const std::string& path, uint8_t* dst, size_t capacity,
                  int* width, int* height) {
  int channels = 0;
  stbi_uc* pixels = stbi_load(path.c_str(), width, height, &channels, 3);
  if (pixels == nullptr) return false;
  const size_t bytes = static_cast<size_t>(*width) * (*height) * 3;
  if (bytes > capacity) {
    stbi_image_free(pixels);
    return false;
  }
  memcpy(dst, pixels, bytes);
  stbi_image_free(pixels);
  return true;
}

bool WriteGrayPng(const std::string& path, const uint8_t* pixels, int width,
                  int height) {
  return stbi_write_png(path.c_str(), width, height, 1, pixels, width) != 0;
}

bool MakeDirectories(const std::string& directory) {
  if (directory.empty()) return true;
  std::string partial;
  size_t start = 0;
  while (start <= directory.size()) {
    size_t slash = directory.find('/', start);
    if (slash == std::string::npos) slash = directory.size();
    partial = directory.substr(0, slash);
    if (!partial.empty() && mkdir(partial.c_str(), 0755) != 0 &&
        errno != EEXIST) {
      return false;
    }
    start = slash + 1;
  }
  return true;
}

std::string BaseNameWithoutExtension(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  if (dot != std::string::npos) name = name.substr(0, dot);
  return name;
}

}  // namespace edge
