# Deployment Guide

## Quick Start

### Prerequisites Check
```bash
# Verify CUDA installation
nvcc --version

# Verify GPU compatibility  
nvidia-smi

# Check compute capability (should be 8.6+ for RTX 4090)
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Build and Test
```bash
# Clone or download the project
cd gpu-vanity

# Build everything
make all

# Run validation tests
make test

# Test with simple prefix
make run
```

## Production Deployment

### 1. Mathematical Validation Required ⚠️

The current implementation contains **placeholder code** that must be replaced:

```cpp
// CRITICAL: These functions need proper implementation
__device__ inline void fe_invert(fe_t &out, const fe_t &a) {
    // Current: Simplified placeholder
    // Required: Full Fermat's little theorem implementation
}

// CRITICAL: Precomputed table is empty
ge_ext h_precomp[TABLE_SIZE];
memset(h_precomp, 0, sizeof(h_precomp));
// Required: Actual basepoint multiples [1G, 2G, ..., 15G]
```

### 2. Security Hardening

```cpp
// CRITICAL: Replace weak nonce-based scalar generation
uint8_t scalar[32];
memset(scalar, 0, 32);
for (int i = 0; i < 8; ++i) {
    scalar[i] = (nonce >> (8*i)) & 0xFF;  // INSECURE
}
```

**Required**: Proper SHA512-based derivation:
```cpp
// Secure approach (pseudocode):
// sha512_hash = SHA512(job_seed || nonce)
// scalar = clamp(sha512_hash[0:32])
```

### 3. Performance Optimization

#### GPU Configuration
```bash
# Set GPU performance mode
sudo nvidia-smi -pm 1

# Set maximum power limit (adjust for your card)
sudo nvidia-smi -pl 450

# Set memory and GPU clocks to maximum
sudo nvidia-smi -ac 10501,2100
```

#### Kernel Tuning
```cpp
// Adjust for your specific GPU
constexpr int THREADS_PER_BLOCK = 256;  // Tune: 128, 256, 512
constexpr int BATCH_K = 8;              // Tune: 4, 8, 16, 32
int blocks = 1024;                      // Tune based on SM count
```

### 4. Multi-GPU Setup

```cpp
// Extend main() for multiple GPUs
int num_devices;
cudaGetDeviceCount(&num_devices);

for (int dev = 0; dev < num_devices; dev++) {
    cudaSetDevice(dev);
    // Launch kernel on each GPU with different nonce ranges
    uint64_t dev_nonce_base = base_nonce + dev * range_per_gpu;
    vanity_kernel<<<blocks, threads>>>(dev_nonce_base, ...);
}
```

### 5. Container Deployment

```dockerfile
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . /app
WORKDIR /app

# Build
RUN make all

# Runtime
ENTRYPOINT ["./sol_vanity_gpu"]
```

Build and run:
```bash
docker build -t solana-vanity-gpu .
docker run --gpus all solana-vanity-gpu "YourPrefix"
```

### 6. Production Checklist

- [ ] **Replace all placeholder math functions**
- [ ] **Implement proper SHA512 scalar derivation**
- [ ] **Generate and validate precomputed basepoint table**
- [ ] **Run comprehensive test suite against reference implementation**
- [ ] **Benchmark performance and tune parameters**
- [ ] **Implement secure private key storage/encryption**
- [ ] **Add monitoring and logging**
- [ ] **Set up proper error handling and recovery**

### 7. Expected Performance

| GPU | Keys/Second | Notes |
|-----|-------------|-------|
| RTX 4090 | ~100M+ | Optimized implementation |
| RTX 4080 | ~80M+ | Slightly reduced SM count |
| RTX 3090 | ~60M+ | Older architecture |

Performance scales exponentially with prefix length:
- 3 chars: ~1 minute
- 4 chars: ~1 hour  
- 5 chars: ~2-3 days
- 6+ chars: weeks+

### 8. Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor performance
./sol_vanity_gpu "ABC" 2>&1 | tee performance.log
```

### 9. Security Considerations

1. **Private Key Protection**: Keys are displayed in plain text
2. **Memory Security**: GPU memory may be accessible to other processes
3. **Side Channel Attacks**: Timing attacks possible in shared environments
4. **Randomness Quality**: Current nonce-based approach is deterministic

### 10. Legal and Compliance

- Verify compliance with local cryptocurrency regulations
- Understand implications of generating addresses for others
- Consider rate limiting and abuse prevention
- Implement proper audit logging

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or threads per block
# Modify BATCH_K and THREADS_PER_BLOCK constants
```

**No matches found**
```bash
# Increase search space or reduce prefix length
# Check Base58 alphabet compatibility
```

**Performance lower than expected**
```bash
# Check GPU clocks: nvidia-smi -q
# Profile with: nvprof ./sol_vanity_gpu "ABC"  
# Optimize memory access patterns
```

For production deployment, thorough code review and mathematical validation by cryptographic experts is strongly recommended.