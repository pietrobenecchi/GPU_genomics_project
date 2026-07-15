cmake_minimum_required(VERSION 3.14)

project(theseus_proxy VERSION 1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(THESEUS_PROXY_BUILD_APP "Build the seq2graph proxy app" ON)

add_library(theseus_proxy
  src/alignment.cpp
  src/penalties.cpp
  src/gfa_graph.cpp
  src/graph.cpp
  src/scope.cpp
  src/theseus_aligner.cpp
  src/theseus_aligner_impl.cpp
)

target_include_directories(theseus_proxy
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(THESEUS_PROXY_BUILD_APP)
  add_executable(seq2graph_proxy apps/seq2graph_proxy.cpp)
  target_link_libraries(seq2graph_proxy PRIVATE theseus_proxy)
  set_target_properties(seq2graph_proxy PROPERTIES
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
  endif()
endif()
