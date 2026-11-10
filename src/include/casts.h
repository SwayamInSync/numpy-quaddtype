#ifndef _QUADDTYPE_CASTS_H
#define _QUADDTYPE_CASTS_H

#include <Python.h>
#include "numpy/dtype_api.h"
#include "quad_common.h"

#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **
init_casts(void);

void
free_casts(void);

quad_value
quad_convert_backend(const quad_value *in, QuadBackendType from, QuadBackendType to);

#ifdef __cplusplus
}
#endif

#endif