# 3.18 e' la prima versione che conosce CUDA_ARCHITECTURES.
cmake_minimum_required(VERSION 3.18)

project(theseus_proxy VERSION 1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(THESEUS_PROXY_BUILD_APP "Build the seq2graph proxy app" ON)
option(THESEUS_ENABLE_CUDA "Enable the CUDA alignment backend" OFF)
option(THESEUS_PROXY_ENABLE_CUDA "Build the CUDA GPU backend (requires nvcc)" ${THESEUS_ENABLE_CUDA})

# Solo src/gpu/*.cu passa da nvcc, il resto resta col compilatore host e i due si
# incontrano su src/gpu/align_gpu.h, di soli POD. Il codice device sta tutto in
# align_gpu.cu, una translation unit sola: i registri dipendono dall'inlining.
if(THESEUS_PROXY_ENABLE_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      # "native" prende la GPU presente al build, e da qui in poi esiste.
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      # CMake piu' vecchio non sa interrogare il device: si coprono le schede
      # probabili. Si sovrascrive con -DCMAKE_CUDA_ARCHITECTURES=<arch>.
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
  # I .cu includono solo l'header di POD, quindi non gli serve il C++20 e nvcc si
  # risparmia le parti dello standard che regge peggio.
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

    # Il backend gpu deve restare byte-identico al baseline CPU, sia quando
    # ricade sia col kernel vero. Vale anche senza CUDA: copre l'entry point del
    # batch e l'appiattimento.
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
