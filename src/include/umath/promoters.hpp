#ifndef _QUADDTYPE_PROMOTERS
#define _QUADDTYPE_PROMOTERS

#include <Python.h>
#include <cstdio>
#include <cassert>
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

#include "../dtype.h"

inline bool
quad_ufunc_has_object_input(PyUFuncObject *ufunc, PyArray_DTypeMeta *const op_dtypes[])
{
    for (int i = 0; i < ufunc->nin; i++) {
        if (op_dtypes[i] == &PyArray_ObjectDType) {
            return true;
        }
    }
    return false;
}

inline int
quad_add_promoter(PyObject *ufunc, PyObject *promoter, PyArray_DTypeMeta *const dtypes[])
{
    int nargs = ((PyUFuncObject *)ufunc)->nargs;
    PyObject *dtype_tuple = PyTuple_New(nargs);
    if (dtype_tuple == NULL) {
        return -1;
    }
    for (int i = 0; i < nargs; i++) {
        Py_INCREF(dtypes[i]);
        PyTuple_SET_ITEM(dtype_tuple, i, (PyObject *)dtypes[i]);
    }

    int res = PyUFunc_AddPromoter(ufunc, dtype_tuple, promoter);
    Py_DECREF(dtype_tuple);
    return res;
}

inline void
quad_set_promoted_dtype(PyArray_DTypeMeta *signature_dtype,
                        PyArray_DTypeMeta *fallback_dtype,
                        PyArray_DTypeMeta **new_dtype)
{
    PyArray_DTypeMeta *dtype = signature_dtype != NULL ? signature_dtype : fallback_dtype;
    Py_INCREF(dtype);
    *new_dtype = dtype;
}

inline int
quad_ufunc_promoter(PyObject *ufunc_obj, PyArray_DTypeMeta *const op_dtypes[],
                    PyArray_DTypeMeta *const signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
    int nargs = ufunc->nargs;

    // Handle the special case for reductions
    if (op_dtypes[0] == NULL) {
        assert(ufunc->nin == 2 && ufunc->nout == 1); /* must be reduction */
        for (int i = 0; i < 3; i++) {
            Py_INCREF(op_dtypes[1]);
            new_op_dtypes[i] = op_dtypes[1];
        }
        return 0;
    }

    PyArray_DTypeMeta *common_dtype = signature[ufunc->nin];
    for (int i = ufunc->nin + 1; i < nargs; i++) {
        if (signature[i] != common_dtype) {
            common_dtype = NULL;
            break;
        }
    }
    if (common_dtype == NULL) {
        common_dtype = quad_ufunc_has_object_input(ufunc, op_dtypes)
                               ? &PyArray_ObjectDType : &QuadPrecDType;
    }
    for (int i = 0; i < nargs; i++) {
        quad_set_promoted_dtype(signature[i], common_dtype, &new_op_dtypes[i]);
    }
    return 0;
}

inline int
quad_add_promoters(PyObject *ufunc_obj)
{
    PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
    assert(ufunc->nin >= 1 && ufunc->nin <= 2 && ufunc->nargs <= 4);
    PyObject *capsule = PyCapsule_New((void *)&quad_ufunc_promoter,
                                    "numpy._ufunc_promoter", NULL);
    if (capsule == NULL) {
        return -1;
    }
    PyArray_DTypeMeta *any_dtype = (PyArray_DTypeMeta *)&PyArrayDescr_Type;
    PyArray_DTypeMeta *pattern[4] = {any_dtype, any_dtype, any_dtype, any_dtype};
    for (int i = 0; i < ufunc->nin; i++) {
        pattern[i] = &QuadPrecDType;
    }

    // All-Quad inputs must precede mixed inputs to avoid ambiguous promotion
    // when an explicit output dtype excludes the Quad loop.
    int res = quad_add_promoter(ufunc_obj, capsule, pattern);
    for (int quad_slot = 0; res == 0 && ufunc->nin == 2 && quad_slot < 2; quad_slot++) {
        pattern[quad_slot] = &QuadPrecDType;
        pattern[1 - quad_slot] = any_dtype;
        res = quad_add_promoter(ufunc_obj, capsule, pattern);
    }
    Py_DECREF(capsule);
    return res;
}


inline int
quad_ldexp_promoter(PyObject *ufunc_obj, PyArray_DTypeMeta *const op_dtypes[],
                    PyArray_DTypeMeta *const signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(&QuadPrecDType);
    new_op_dtypes[0] = &QuadPrecDType;

    // Promote the exponent to PyArray_IntpDType (unless signature specifies otherwise)
    if (signature[1] != NULL) {
        Py_INCREF(signature[1]);
        new_op_dtypes[1] = signature[1];
    }
    else {
        Py_INCREF(&PyArray_IntpDType);
        new_op_dtypes[1] = &PyArray_IntpDType;
    }

    Py_INCREF(&QuadPrecDType);
    new_op_dtypes[2] = &QuadPrecDType;

    return 0;
}

#endif
