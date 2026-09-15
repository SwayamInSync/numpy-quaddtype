#ifndef _QUADDTYPE_DTYPE_H
#define _QUADDTYPE_DTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/dtype_api.h>
#include "quad_common.h"

typedef struct {
    PyArray_Descr base;
    QuadBackendType backend;
} QuadPrecDTypeObject;

extern PyArray_DTypeMeta QuadPrecDType;

QuadPrecDTypeObject *
new_quaddtype_instance(QuadBackendType backend);

/* Fail a resolver; `loop_descrs` must be NULL-initialized by the caller. */
static inline NPY_CASTING
quad_resolve_descrs_fail(PyArray_Descr *loop_descrs[], int n)
{
    for (int i = 0; i < n; i++) {
        Py_CLEAR(loop_descrs[i]);
    }
    return (NPY_CASTING)-1;
}

int
init_quadprec_dtype(void);

#ifdef __cplusplus
}
#endif

#endif