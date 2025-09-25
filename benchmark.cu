/*
benchmark.cu - Performance benchmarking for GPU vanity generator
*/

#include <cstdio>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// Configuration for benchmarking
constexpr int BENCHMARK_ITERATIONS = 10;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int BATCH_K = 8;
constexpr uint64_t KEYS_PER_ITERATION = 1024 * THREADS_PER_BLOCK * BATCH_K;

// Minimal kernel for throughput testing (without actual crypto operations)
__global__ void benchmark_kernel(uint64_t nonce_base, uint64_t *results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t my_base = nonce_base + (uint64_t)tid * (uint64_t)BATCH_K;
    
    uint64_t operations = 0;
    
    for (int k = 0; k < BATCH_K; ++k) {
        uint64_t nonce = my_base + (uint64_t)k;
        
        // Simulate key generation work (simplified operations)
        for (int i = 0; i < 100; ++i) {  // Simulate computation load
            nonce = nonce * 1103515245ULL + 12345ULL;  // Simple LCG
            operations++;
        }
        
        // Simulate prefix check
        if ((nonce & 0xFFFFFF) == 0x123456) {  // Rare condition
            atomicAdd((unsigned long long*)&results[0], 1ULL);
        }
    }
    
    // Count total operations performed
    atomicAdd((unsigned long long*)&results[1], (unsigned long long)operations);
}

class GPUBenchmark {
private:
    cudaDeviceProp device_props;
    int device_id;
    
public:
    GPUBenchmark() {
        cudaSetDevice(0);
        cudaGetDevice(&device_id);
        cudaGetDeviceProperties(&device_props, device_id);
    }
    
    void print_device_info() {
        printf("=== GPU Benchmark ===\n");
        printf("Device: %s\n", device_props.name);
        printf("Compute Capability: %d.%d\n", device_props.major, device_props.minor);
        printf("Multiprocessors: %d\n", device_props.multiProcessorCount);
        printf("Max Threads per Block: %d\n", device_props.maxThreadsPerBlock);
        printf("Global Memory: %.1f GB\n", (double)device_props.totalGlobalMem / (1024*1024*1024));
        printf("Memory Clock: %d MHz\n", device_props.memoryClockRate / 1000);
        printf("Memory Bus Width: %d-bit\n", device_props.memoryBusWidth);
        printf("\n");
    }
    
    void run_throughput_test() {
        printf("=== Throughput Test ===\n");
        printf("Configuration:\n");
        printf("  Threads per Block: %d\n", THREADS_PER_BLOCK);
        printf("  Batch Size: %d\n", BATCH_K);
        printf("  Keys per Iteration: %llu\n", (unsigned long long)KEYS_PER_ITERATION);
        printf("\n");
        
        uint64_t *d_results, *h_results;
        cudaMalloc(&d_results, 2 * sizeof(uint64_t));
        h_results = new uint64_t[2];
        
        // Warm up
        cudaMemset(d_results, 0, 2 * sizeof(uint64_t));
        benchmark_kernel<<<1024, THREADS_PER_BLOCK>>>(0, d_results);
        cudaDeviceSynchronize();
        
        printf("Running benchmark...\n");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < BENCHMARK_ITERATIONS; ++iter) {
            cudaMemset(d_results, 0, 2 * sizeof(uint64_t));
            
            uint64_t nonce_base = iter * KEYS_PER_ITERATION;
            benchmark_kernel<<<1024, THREADS_PER_BLOCK>>>(nonce_base, d_results);
            cudaDeviceSynchronize();
            
            if (iter % (BENCHMARK_ITERATIONS / 4) == 0) {
                printf("  Iteration %d/%d completed\n", iter + 1, BENCHMARK_ITERATIONS);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Get final results
        cudaMemcpy(h_results, d_results, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        uint64_t total_keys = BENCHMARK_ITERATIONS * KEYS_PER_ITERATION;
        double seconds = duration.count() / 1000.0;
        double keys_per_second = total_keys / seconds;
        double million_keys_per_second = keys_per_second / 1000000.0;
        
        printf("\nResults:\n");
        printf("  Total Keys Processed: %llu\n", (unsigned long long)total_keys);
        printf("  Time Elapsed: %.3f seconds\n", seconds);
        printf("  Throughput: %.2f million keys/second\n", million_keys_per_second);
        printf("  Memory Bandwidth Utilization: %.1f%%\n", estimate_memory_bandwidth_usage(keys_per_second));
        printf("  Compute Utilization: %.1f%%\n", estimate_compute_usage());
        
        cleanup(d_results, h_results);
    }
    
    void run_memory_test() {
        printf("\n=== Memory Performance Test ===\n");
        
        size_t test_size = 256 * 1024 * 1024;  // 256 MB
        uint8_t *d_data, *h_data;
        
        cudaMalloc(&d_data, test_size);
        h_data = new uint8_t[test_size];
        
        // Fill test data
        for (size_t i = 0; i < test_size; ++i) {
            h_data[i] = i & 0xFF;
        }
        
        printf("Testing memory transfer speeds...\n");
        
        // Host to Device
        auto start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_data, h_data, test_size, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto h2d_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double h2d_bandwidth = (test_size / (1024.0 * 1024.0 * 1024.0)) / (h2d_time.count() / 1000.0);
        
        // Device to Host
        start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_data, d_data, test_size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        
        auto d2h_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double d2h_bandwidth = (test_size / (1024.0 * 1024.0 * 1024.0)) / (d2h_time.count() / 1000.0);
        
        printf("  Host to Device: %.2f GB/s\n", h2d_bandwidth);
        printf("  Device to Host: %.2f GB/s\n", d2h_bandwidth);
        
        cudaFree(d_data);
        delete[] h_data;
    }
    
private:
    double estimate_memory_bandwidth_usage(double keys_per_second) {
        // Estimate based on memory accesses per key operation
        double bytes_per_key = 64;  // Rough estimate
        double estimated_bandwidth = keys_per_second * bytes_per_key / (1024.0 * 1024.0 * 1024.0);
        
        // Peak theoretical bandwidth (example for RTX 4090: ~1000 GB/s)
        double peak_bandwidth = device_props.memoryBusWidth / 8.0 * device_props.memoryClockRate * 2 / 1000000;
        
        return (estimated_bandwidth / peak_bandwidth) * 100.0;
    }
    
    double estimate_compute_usage() {
        // Simplified estimate based on SM utilization
        int blocks_per_sm = device_props.maxThreadsPerMultiProcessor / THREADS_PER_BLOCK;
        int total_blocks = 1024;  // From kernel launch
        int active_sms = std::min(total_blocks / blocks_per_sm, device_props.multiProcessorCount);
        
        return (double(active_sms) / device_props.multiProcessorCount) * 100.0;
    }
    
    void cleanup(uint64_t *d_results, uint64_t *h_results) {
        cudaFree(d_results);
        delete[] h_results;
    }
};

int main() {
    GPUBenchmark benchmark;
    
    benchmark.print_device_info();
    benchmark.run_throughput_test();
    benchmark.run_memory_test();
    
    printf("\n=== Benchmark Complete ===\n");
    printf("Note: This benchmark simulates computational load but does not\n");
    printf("perform actual Ed25519 operations. Real performance will depend\n");
    printf("on the complexity of cryptographic operations.\n");
    
    return 0;
}