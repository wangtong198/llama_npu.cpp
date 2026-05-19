message(STATUS "Using AWNPU backend")

file(GLOB GGML_HEADERS_AWNPU "*.h")
file(GLOB GGML_SOURCES_AWNPU "*.cpp")

ggml_add_backend_library(ggml-awnpu
    ${GGML_HEADERS_AWNPU}
    ${GGML_SOURCES_AWNPU}
)
