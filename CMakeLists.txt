cmake_minimum_required(VERSION 3.18)
project(SolanaVanityGPU LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures for RTX 4090
set(CMAKE_CUDA_ARCHITECTURES 86)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# Add executable
add_executable(sol_vanity_gpu sol_vanity_gpu.cu)

# Set target properties
set_target_properties(sol_vanity_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(sol_vanity_gpu ${CUDA_LIBRARIES})

# Install target
install(TARGETS sol_vanity_gpu DESTINATION bin)