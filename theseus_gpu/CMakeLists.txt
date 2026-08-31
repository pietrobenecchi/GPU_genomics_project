# 3.18 is the first version that knows about CUDA_ARCHITECTURES.
cmake_minimum_required(VERSION 3.18)

project(theseus_proxy VERSION 1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(THESEUS_PROXY_BUILD_APP "Build the seq2graph proxy app" ON)
option(THESEUS_ENABLE_CUDA "Enable the CUDA alignment backend" OFF)
option(THESEUS_PROXY_ENABLE_CUDA "Build the CUDA GPU backend (requires nvcc)" ${THESEUS_ENABLE_CUDA})

# Only src/gpu/*.cu goes through nvcc: the rest of the library keeps using the
# host compiler, and the two meet at the POD-only src/gpu/align_gpu.h. Of the
# four, align_gpu.cu is the one holding device code -- it is a single
# translation unit on purpose, because the kernel's register count depends on
# nvcc inlining the whole call chain into it. The other three are host code that
# reaches the kernels through src/gpu/kernel_launch.h.
if(THESEUS_PROXY_ENABLE_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      # "native" targets the GPU present at build time, and only exists here.
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      # Older CMake cannot probe the device: cover the cards this is likely to
      # run on. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> if yours differs.
      set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
  endif()
  message(STATUS "theseus_proxy: CUDA backend ON, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  set(THESEUS_PROXY_GPU_SOURCES
    src/gpu/align_gpu.cu
    src/gpu/align_batch.cu
    src/gpu/device_memory.cu
    src/gpu/gpu_error.cu)
else()
  set(THESEUS_PROXY_GPU_SOURCES src/gpu/align_gpu_stub.cpp)
endif()

add_library(theseus_proxy
  src/alignment.cpp
  src/penalties.cpp
  src/gfa_graph.cpp
  src/graph.cpp
  src/scope.cpp
  src/theseus_aligner.cpp
  src/theseus_aligner_impl.cpp
  src/gpu/graph_csr.cpp
  ${THESEUS_PROXY_GPU_SOURCES}
)

target_include_directories(theseus_proxy
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(THESEUS_PROXY_ENABLE_CUDA)
  # The .cu files only include the POD header, so they do not need C++20 and
  # nvcc is spared the parts of the standard it handles worst.
  set_target_properties(theseus_proxy PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
endif()

if(THESEUS_PROXY_BUILD_APP)
  add_executable(seq2graph_proxy apps/seq2graph_proxy.cpp)
  target_link_libraries(seq2graph_proxy PRIVATE theseus_proxy)
  set_target_properties(seq2graph_proxy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps"
  )

  add_executable(seq2graph_gpu_validate apps/seq2graph_gpu_validate.cpp)
  target_link_libraries(seq2graph_gpu_validate PRIVATE theseus_proxy)
  set_target_properties(seq2graph_gpu_validate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps"
  )

  add_executable(seq2graph_gpu_benchmark apps/seq2graph_gpu_benchmark.cpp)
  target_link_libraries(seq2graph_gpu_benchmark PRIVATE theseus_proxy)
  set_target_properties(seq2graph_gpu_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps"
  )


  if(BUILD_TESTING)
    add_test(
      NAME cpu_sample_baseline_run
      COMMAND $<TARGET_FILE:seq2graph_proxy>
        -g ${CMAKE_CURRENT_SOURCE_DIR}/data/sample_graph.gfa
        -s ${CMAKE_CURRENT_SOURCE_DIR}/data/sample_queries.fasta
        -f ${CMAKE_BINARY_DIR}/cpu_sample_output.gaf
    )

    add_test(
      NAME cpu_sample_baseline_compare
      COMMAND ${CMAKE_COMMAND} -E compare_files
        ${CMAKE_CURRENT_SOURCE_DIR}/baseline/sample_output.gaf
        ${CMAKE_BINARY_DIR}/cpu_sample_output.gaf
    )
    set_tests_properties(cpu_sample_baseline_compare PROPERTIES
      DEPENDS cpu_sample_baseline_run
    )

    # The gpu backend must keep matching the CPU baseline byte for byte, both
    # while it falls back and once the kernel lands. Worth running even without
    # CUDA: it covers the batch entry point and the flattening.
    add_test(
      NAME gpu_sample_baseline_run
      COMMAND $<TARGET_FILE:seq2graph_proxy>
        -b gpu
        -g ${CMAKE_CURRENT_SOURCE_DIR}/data/sample_graph.gfa
        -s ${CMAKE_CURRENT_SOURCE_DIR}/data/sample_queries.fasta
        -f ${CMAKE_BINARY_DIR}/gpu_sample_output.gaf
    )

    add_test(
      NAME gpu_sample_baseline_compare
      COMMAND ${CMAKE_COMMAND} -E compare_files
        ${CMAKE_CURRENT_SOURCE_DIR}/baseline/sample_output.gaf
        ${CMAKE_BINARY_DIR}/gpu_sample_output.gaf
    )
    set_tests_properties(gpu_sample_baseline_compare PROPERTIES
      DEPENDS gpu_sample_baseline_run
    )


    add_test(
      NAME gpu_cpu_validation
      COMMAND $<TARGET_FILE:seq2graph_gpu_validate>
    )
  endif()
endif()
