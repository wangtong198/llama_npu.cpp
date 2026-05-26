message(STATUS "Using AWNPU backend")

set(GGML_SOURCES_AWNPU
    ggml-awnpu.cpp
    ggml-awnpu-graph-node.cpp
    ggml-awnpu-layer-map.cpp
)

if (KERNEL_TYPE STREQUAL "native")
    message(STATUS "AWNPU: using native CPU reference op kernels implementated by pure C/C++")
    list(APPEND GGML_SOURCES_AWNPU ggml-awnpu-kernels-native.cpp)
elseif (KERNEL_TYPE STREQUAL "awnpu")
    message(STATUS "AWNPU: using awnpu op kernels")
    list(APPEND GGML_SOURCES_AWNPU ggml-awnpu-kernels-npu.cpp)
else()
    message(FATAL_ERROR "AWNPU: unsupported KERNEL_TYPE='${KERNEL_TYPE}' (expected awnpu or native)")
endif()

set(GGML_HEADERS_AWNPU
    ggml-awnpu-graph-node.h
    ggml-awnpu-kernels-native.h
    ggml-awnpu-kernels-npu.h
    ggml-awnpu-layer-map.h
)

ggml_add_backend_library(ggml-awnpu
    ${GGML_HEADERS_AWNPU}
    ${GGML_SOURCES_AWNPU}
)

target_include_directories(ggml-awnpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

if (KERNEL_TYPE STREQUAL "native")
    target_compile_definitions(ggml-awnpu PRIVATE GGML_AWNPU_KERNEL_TYPE_NATIVE)
elseif (KERNEL_TYPE STREQUAL "awnpu")
    target_compile_definitions(ggml-awnpu PRIVATE GGML_AWNPU_KERNEL_TYPE_AWNPU)
endif()

if (KERNEL_TYPE STREQUAL "native")
    # Correctness test: compare native kernel outputs against CPU reference.
    add_executable(test-awnpu-kernels test-awnpu-kernels.cpp)
    target_include_directories(test-awnpu-kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..    # for ggml-impl.h (ggml/src/)
    )
    target_link_libraries(test-awnpu-kernels PRIVATE ggml-awnpu ggml-cpu ggml-base)
    target_compile_features(test-awnpu-kernels PRIVATE cxx_std_17)
endif()
