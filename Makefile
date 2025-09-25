# Makefile for Solana GPU Vanity Address Generator

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -std=c++17 --expt-relaxed-constexpr
CUDA_ARCH = sm_86  # RTX 4090 architecture
GENCODE = -gencode arch=compute_86,code=sm_86

# Target executables
TARGET = sol_vanity_gpu
COMPLETE = sol_vanity_complete
VALIDATE = validate_gpu
BENCHMARK = benchmark_gpu
PRECOMPUTE = precompute

# Source files
SOURCES = sol_vanity_gpu.cu
COMPLETE_SOURCES = sol_vanity_complete.cu
VALIDATE_SOURCES = validate.cu
BENCHMARK_SOURCES = benchmark.cu
PRECOMPUTE_SOURCES = precompute.cu

# Build rules
all: $(TARGET) $(COMPLETE) $(VALIDATE) $(BENCHMARK) $(PRECOMPUTE)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -o $(TARGET) $(SOURCES)

$(COMPLETE): $(COMPLETE_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -o $(COMPLETE) $(COMPLETE_SOURCES)

$(VALIDATE): $(VALIDATE_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -o $(VALIDATE) $(VALIDATE_SOURCES)

$(BENCHMARK): $(BENCHMARK_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -o $(BENCHMARK) $(BENCHMARK_SOURCES)

$(PRECOMPUTE): $(PRECOMPUTE_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -o $(PRECOMPUTE) $(PRECOMPUTE_SOURCES)

# Clean rule
clean:
	rm -f $(TARGET) $(COMPLETE) $(VALIDATE) $(BENCHMARK) $(PRECOMPUTE)

# Test build (compile only)
compile-test:
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -c $(SOURCES)

# Run validation tests
test: $(VALIDATE)
	./$(VALIDATE)

# Install dependencies (if needed)
deps:
	@echo "CUDA toolkit should be installed separately"
	@echo "Verify with: nvcc --version"

# Run skeleton version
run: $(TARGET)
	./$(TARGET) "Sol"

# Run complete production version  
run-complete: $(COMPLETE)
	./$(COMPLETE) "Sol"

# Generate precomputed table
precomp: $(PRECOMPUTE)
	./$(PRECOMPUTE) > precomputed_table.h

# Run performance benchmark
bench: $(BENCHMARK)
	./$(BENCHMARK)

.PHONY: all clean compile-test test deps run run-complete precomp bench