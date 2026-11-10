# Project description: CUDA Batch Edge Detection

**Purpose.** I built a GPU program that runs edge detection on a large batch of
images in a single execution. One run processes 256 images, 240 small
(256×256) and 16 large (1024×1024), about 32.5 megapixels in total. The
program writes an edge map for every image, a gallery of side-by-side previews,
a per-image log and a summary with throughput and the GPU-vs-CPU speedup.

**Algorithm and kernels.** Each image goes through four CUDA kernels that I
wrote myself:

1. **RGB to grayscale** (BT.601 weights).
2. **Gaussian blur, row pass.**
3. **Gaussian blur, column pass.** Splitting the blur into two 1-D passes
   (separable filter) means each pixel needs 2k reads instead of k² for a k×k
   kernel. For σ = 1.4 that is 22 instead of 121.
4. **Sobel gradient magnitude**, written either as a grayscale magnitude image
   or as a binary edge map (`--threshold`).

The Gaussian weights are stored in `__constant__` memory. All threads in a warp
read the same weight at the same time, which the constant cache handles well.
Borders use clamp-to-edge.

**Scaling to many images.** The batch is spread over several CUDA streams
(`--num_streams`). Each stream has its own pinned host buffers and device
buffers, so copies are truly asynchronous. While the CPU decodes or encodes a
PNG for one stream, the other streams keep the GPU busy. Buffers grow when a
larger image arrives, so small and large images can be mixed in one run. Each
image's GPU time is measured with CUDA events.

**Correctness.** The per-pixel math is written once as `__host__ __device__`
functions and used by both the GPU kernels and a single-threaded CPU reference.
With `--verify`, every GPU output is compared pixel by pixel with the CPU
result, and the log shows the maximum difference for each image.

**Results.** On the lab GPU, all 256 images were processed and verified against
the CPU (see `logs/summary.md`). The summary reports:

- [FILL IN: copy the "Sum of per-image GPU time" value from logs/summary.md] of GPU time for the whole batch
- [FILL IN: copy the "GPU speedup vs. 1 CPU core" value from logs/summary.md] speedup over one CPU core
- [FILL IN: copy the "Throughput (wall clock)" value from logs/summary.md] end-to-end throughput

**Challenges and lessons learned.**

- **GPU and CPU floating point differ.** By default nvcc fuses `a*b + c` into
  one FMA instruction, which rounds differently from the CPU. That can shift a
  pixel by one gray level and make a strict comparison fail even though both
  results are "correct." Building with `--fmad=false` and sharing the same
  per-pixel functions made the GPU and CPU results identical, so the
  verification can be exact.
- **GPU time vs. total time.** The log reports both the wall-clock time
  (which includes decoding and encoding PNGs on the CPU) and the sum of the
  per-image GPU times. In my run these were [FILL IN: copy the "Wall time" value
  from logs/summary.md] and [FILL IN: copy the "Sum of per-image GPU time"
  value]. The difference shows how much of the run is file I/O rather than GPU
  work. This is why streams matter here: they let the GPU keep working while
  the CPU is busy with files.
- **Generality.** Supporting mixed image sizes, unreadable files and any
  block-size or stream-count combination meant validating the arguments and
  managing buffer sizes carefully, rather than assuming one fixed image size.
- **Next steps.** I would add shared-memory tiling for the blur and Sobel
  stages, a multi-threaded image loader so that file I/O keeps up with the GPU,
  and NPP or nvJPEG for decoding images on the GPU.

**How to run:** `./run.sh` (it builds the code, generates the dataset and runs
the batch). All options are listed in `README.md`.
