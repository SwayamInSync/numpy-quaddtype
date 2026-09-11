import numpy as np
import pytest

from numpy_quaddtype import QuadPrecDType

SAME_DTYPE_UFUNCS = [
    np.add, np.subtract, np.multiply, np.divide, np.floor_divide, np.power,
    np.remainder, np.minimum, np.maximum, np.fmin, np.fmax, np.matmul, np.fmod,
    np.arctan2, np.hypot, np.float_power, np.copysign, np.nextafter, np.logaddexp,
    np.logaddexp2, np.heaviside, np.divmod, np.absolute, np.negative, np.positive,
    np.sign, np.sqrt, np.square, np.cbrt, np.reciprocal, np.exp, np.exp2, np.expm1,
    np.log, np.log2, np.log10, np.log1p, np.sin, np.cos, np.tan, np.arcsin,
    np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh,
    np.arctanh, np.floor, np.ceil, np.trunc, np.rint, np.fabs, np.spacing,
    np.conjugate, np.degrees, np.radians, np.deg2rad, np.rad2deg, np.modf,
]


@pytest.fixture(params=["sleef", "longdouble"])
def quad(request):
    return np.array(
        [[1.5, 2.5], [3.5, 4.5]], dtype=QuadPrecDType(backend=request.param)
    )


def assert_outputs_equal(ufunc, actual, expected):
    if ufunc.nout == 1:
        actual, expected = (actual,), (expected,)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference, strict=True)


@pytest.mark.parametrize("ufunc", SAME_DTYPE_UFUNCS)
@pytest.mark.parametrize("keyword", ["dtype", "signature"])
def test_explicit_float64_matches_numpy(quad, ufunc, keyword):
    kwargs = {"dtype": np.float64}
    if keyword == "signature":
        kwargs = {"signature": (None,) * ufunc.nin + (np.float64,) * ufunc.nout}
    with np.errstate(all="ignore"):
        actual = ufunc(*(quad,) * ufunc.nin, **kwargs)
        expected = ufunc(*(quad.astype(np.float64),) * ufunc.nin, **kwargs)

    assert_outputs_equal(ufunc, actual, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_precision_matches_float32_float64(quad, reverse):
    higher = np.array([2**24 + 1], dtype=quad.dtype)
    lower = np.array([-2**24], dtype=np.float32)
    args = (lower, higher) if reverse else (higher, lower)
    reference_args = tuple(a.astype(np.float64) if a is higher else a for a in args)

    result = np.add(*args)
    assert result.dtype == quad.dtype
    np.testing.assert_array_equal(
        result.astype(np.float64), np.add(*reference_args), strict=True
    )

    rounded = np.add(*args, dtype=np.float32)
    np.testing.assert_array_equal(
        rounded, np.add(*reference_args, dtype=np.float32), strict=True
    )
    np.testing.assert_array_equal(rounded, [0])

    out = np.empty(1, dtype=np.float32)
    reference_out = np.empty_like(out)
    assert np.add(*args, out=out) is out
    np.add(*reference_args, out=reference_out)
    np.testing.assert_array_equal(out, reference_out, strict=True)
    np.testing.assert_array_equal(out, [1])


@pytest.mark.parametrize("dtype, casting", [
    (np.float32, "same_kind"), (np.float64, "unsafe"),
    (np.int64, "unsafe"), (np.bool_, "unsafe"), (object, "safe"),
])
def test_explicit_dtype_allows_casts(quad, dtype, casting):
    reference = quad.astype(object if dtype is object else np.float64)
    actual = np.add(quad, quad, dtype=dtype, casting=casting)
    expected = np.add(reference, reference, dtype=dtype, casting=casting)
    np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.parametrize("dtype, casting", [
    (np.float32, "no"), (np.float32, "equiv"), (np.float32, "safe"),
    (np.int64, "same_kind"), (np.bool_, "same_kind"),
])
def test_explicit_dtype_rejects_casts(quad, dtype, casting):
    for operand in [quad, quad.astype(np.float64)]:
        with pytest.raises(TypeError):
            np.add(operand, operand, dtype=dtype, casting=casting)


def test_explicit_dtype_requires_registered_cast(quad):
    with pytest.raises(TypeError):
        np.add(quad, quad, dtype=np.complex128, casting="unsafe")


@pytest.mark.parametrize("ufunc", [np.add, np.negative, np.matmul])
def test_explicit_object_dtype(quad, ufunc):
    actual = ufunc(*(quad,) * ufunc.nin, dtype=object)
    expected = ufunc(*(quad.astype(object),) * ufunc.nin)
    np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.parametrize("ufunc, error", [
    (np.sqrt, TypeError), (np.fmod, AttributeError),
    (np.modf, TypeError), (np.divmod, TypeError),
])
def test_explicit_object_dtype_without_supported_loop(quad, ufunc, error):
    for operand in [quad, quad.astype(object)]:
        with pytest.raises(error):
            ufunc(*(operand,) * ufunc.nin, dtype=object)


@pytest.mark.parametrize("ufunc", [np.add, np.divmod])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("keyword", ["dtype", "signature"])
def test_explicit_float_dtype_overrides_object_promotion(quad, ufunc, reverse, keyword):
    objects = quad.astype(object)
    args = (objects, quad) if reverse else (quad, objects)
    reference_args = tuple(a.astype(np.float64) if a is quad else a for a in args)
    kwargs = {"dtype": np.float64}
    if keyword == "signature":
        kwargs = {"signature": (None, None) + (np.float64,) * ufunc.nout}
    for operands in [args, reference_args]:
        with pytest.raises(TypeError):
            ufunc(*operands, **kwargs)
    actual = ufunc(*args, **kwargs, casting="unsafe")
    expected = ufunc(*reference_args, **kwargs, casting="unsafe")
    assert_outputs_equal(ufunc, actual, expected)


@pytest.mark.parametrize("ufunc", [np.modf, np.divmod])
@pytest.mark.parametrize("outputs", [
    (np.float32, None), (None, np.float32), (np.float32, np.float64),
])
def test_partial_or_conflicting_outputs_match_numpy(quad, ufunc, outputs):
    signature = (None,) * ufunc.nin + outputs
    for operand in [quad, quad.astype(np.float64)]:
        with pytest.raises(TypeError):
            ufunc(*(operand,) * ufunc.nin, signature=signature)


@pytest.mark.parametrize("fixed_input", [0, 1])
def test_fixed_input_is_not_overridden(quad, fixed_input):
    for operand, input_dtype in [
        (quad, QuadPrecDType), (quad.astype(np.float64), np.float64)
    ]:
        signature = [None, None, np.float32]
        signature[fixed_input] = input_dtype
        with pytest.raises(TypeError):
            np.add(operand, operand, signature=tuple(signature))


def test_object_conjugate_preserves_values_and_backend(quad):
    actual = np.conjugate(quad, dtype=object)
    np.testing.assert_array_equal(actual, quad.astype(object), strict=True)
    for scalar in actual.flat:
        assert scalar.dtype == quad.dtype


class TestPromoterNoInterference:
    def test_timedelta_modulus_raises_typeerror(self):
        with pytest.raises(TypeError, match="remainder"):
            np.remainder(np.timedelta64(7, "Y"), 15)

    def test_timedelta_divide_preserves_dtype(self):
        values = np.arange(1000, dtype="m8[s]")
        result = values.sum() / len(values)
        assert result.dtype.kind == "m"

    def test_timedelta_mean_correct(self):
        values = np.arange(1000, dtype="m8[s]")
        np.testing.assert_array_equal(values.mean(), values.sum() / len(values))

    def test_matmul_float64_preserves_values(self):
        identity = np.eye(3)
        values = np.ones((3, 2))
        np.testing.assert_array_equal(np.matmul(identity, values), values, strict=True)

    @pytest.mark.parametrize("ufunc", SAME_DTYPE_UFUNCS)
    def test_builtin_inputs_preserve_dtype(self, ufunc):
        operand = np.array([[1.5, 2.5]], dtype=np.float64)
        args = (operand, operand.T) if ufunc is np.matmul else (operand,) * ufunc.nin
        with np.errstate(all="ignore"):
            result = ufunc(*args)
        for output in result if ufunc.nout == 2 else (result,):
            assert output.dtype == np.float64


@pytest.mark.parametrize("method", ["reduce", "accumulate", "reduceat"])
def test_explicit_float64_reductions(quad, method):
    operand = quad.ravel()
    indices = ([0, 2],) if method == "reduceat" else ()

    actual = getattr(np.add, method)(operand, *indices, dtype=np.float64)
    expected = getattr(np.add, method)(operand.astype(np.float64), *indices)

    np.testing.assert_array_equal(actual, expected, strict=True)
