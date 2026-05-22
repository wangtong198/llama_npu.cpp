#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_AWNPU_NAME "AWNPU"

GGML_API ggml_backend_reg_t ggml_backend_awnpu_reg(void);

#ifdef __cplusplus
}
#endif
