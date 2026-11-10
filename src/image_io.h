// Copyright 2026 Adhiraj Singh
//
// Image file helpers: directory listing, loading (PNG/JPG/BMP/TGA/PGM/PPM via
// stb_image) and writing 8-bit grayscale PNGs (via stb_image_write).

#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace edge {

// Returns the sorted paths of all supported image files in `directory`.
std::vector<std::string> ListImageFiles(const std::string& directory);

// Loads an image as interleaved 8-bit RGB into `dst`, which must have room
// for width * height * 3 bytes. Call ProbeImage first to get the size.
bool ProbeImage(const std::string& path, int* width, int* height);
bool LoadRgbImage(const std::string& path, uint8_t* dst, size_t capacity,
                  int* width, int* height);

// Writes an 8-bit single channel image as PNG.
bool WriteGrayPng(const std::string& path, const uint8_t* pixels, int width,
                  int height);

// Creates `directory` (and parents) if it does not exist.
bool MakeDirectories(const std::string& directory);

// "data/input/foo.png" -> "foo".
std::string BaseNameWithoutExtension(const std::string& path);

}  // namespace edge

#endif  // SRC_IMAGE_IO_H_
