cmake_minimum_required(VERSION 3.24)
project(eonego_cuda LANGUAGES CXX)

add_library(eonego_cuda SHARED eonego_cuda_stub.cpp)
target_compile_features(eonego_cuda PRIVATE cxx_std_17)
target_include_directories(eonego_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
  set_target_properties(eonego_cuda PROPERTIES PREFIX "")
endif()

# Current implementation keeps the final C ABI and uses dynamic CUDA/NVRTC when available, with scalar
# CPU fallback. It intentionally avoids nvcc so the MinGW build works without MSVC cl.exe.
