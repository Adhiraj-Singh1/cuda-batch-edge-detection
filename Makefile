# Build file for the CUDA batch edge detection project.
#
#   make            build bin/edge_detect and bin/generate_dataset
#   make data       generate the synthetic dataset in data/input
#   make run        build and run with ARGS="..." (see ./bin/edge_detect -h)
#   make clean      remove build products
#   make clean-all  also remove generated data, outputs and logs
#
# Set ARCH to target a specific GPU, e.g. make ARCH="-arch=sm_75".

NVCC ?= nvcc
CXX ?= g++
ARCH ?=

INCLUDES := -I.
# --fmad=false keeps GPU floating point bit-compatible with the CPU
# reference so that --verify can do an exact comparison.
NVCCFLAGS := -std=c++17 -O2 --fmad=false -Wno-deprecated-gpu-targets \
             $(INCLUDES) $(ARCH)
CXXFLAGS := -std=c++17 -O2 -Wall $(INCLUDES)

BIN_DIR := bin
OBJ_DIR := obj
TARGET := $(BIN_DIR)/edge_detect
GENERATOR := $(BIN_DIR)/generate_dataset

CU_SRCS := src/main.cu src/edge_pipeline.cu
CC_SRCS := src/image_io.cc src/cpu_reference.cc src/cli_options.cc
HEADERS := $(wildcard src/*.h src/*.cuh)
OBJS := $(CU_SRCS:src/%.cu=$(OBJ_DIR)/%.cu.o) \
        $(CC_SRCS:src/%.cc=$(OBJ_DIR)/%.cc.o)

.PHONY: all build data run clean clean-all

all: build
build: $(TARGET) $(GENERATOR)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

$(OBJ_DIR)/%.cu.o: src/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cc.o: src/%.cc $(HEADERS) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(GENERATOR): tools/generate_dataset.cc | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

# 240 small (256x256) + 16 large (1024x1024) images = 256 inputs, 32 MP.
data: $(GENERATOR)
	mkdir -p data/input
	./$(GENERATOR) -o data/input -c 240 -W 256 -H 256 -p small -s 2026
	./$(GENERATOR) -o data/input -c 16 -W 1024 -H 1024 -p large -s 7

run: build
	./$(TARGET) $(ARGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

clean-all: clean
	rm -rf data/input output logs/run_log.txt logs/summary.md
