"""Windows free-threaded isolation probe for numpy-quaddtype.

Prints the ACTUAL computed values for the operations that fail on the Windows
cp313t (free-threaded) wheel but pass on cp313 (GIL) and on Linux/macOS. Run the
SAME script under both interpreters and diff the ``[GIL]`` vs ``[FT]`` blocks to
see exactly what diverges.

This is a diagnostic, not a gate: it never raises to the shell (always exits 0),
so both wheels' output is captured in the CI log.
"""
import sys

import numpy as np
import numpy_quaddtype as nq
from numpy_quaddtype import QuadPrecision as Q


def gil_enabled():
    f = getattr(sys, "_is_gil_enabled", None)
    return bool(f()) if f is not None else True


def fval(x):
    try:
        return float(x)
    except Exception as e:  # noqa: BLE001
        return "<err %s>" % type(e).__name__


def bits(q):
    try:
        return q.__reduce__()[1][0].hex()
    except Exception:  # noqa: BLE001
        return "?"


def run(label, fn):
    try:
        print("  %-46s -> %s" % (label, fn()))
    except Exception as e:  # noqa: BLE001
        print("  %-46s -> ERROR %s: %s" % (label, type(e).__name__, e))


TAG = "FT" if not gil_enabled() else "GIL"

print("\n########## quaddtype win-ft probe [%s] ##########" % TAG)
print("python    : %s  gil_enabled=%s" % (sys.version.split()[0], gil_enabled()))
print("numpy     : %s" % np.__version__)
print("quaddtype : %s" % getattr(nq, "__version__", "?"))
run("is_longdouble_128", nq.is_longdouble_128)

print("[%s] PURE numpy special-value checks (NO quaddtype involved):" % TAG)
run("np.isinf(float('inf'))", lambda: np.isinf(float("inf")))
run("np.isnan(float('nan'))", lambda: np.isnan(float("nan")))
run("np.isinf(np.inf)", lambda: np.isinf(np.inf))
run("np.isinf(np.float64('inf'))", lambda: np.isinf(np.float64("inf")))
run("np.isnan(np.float64('nan'))", lambda: np.isnan(np.float64("nan")))
run("np.isinf(np.array([np.inf]))[0]", lambda: np.isinf(np.array([np.inf]))[0])
run("bool(np.isinf(1e400))", lambda: bool(np.isinf(1e400)))
run("float(Q('inf'))==float('inf')", lambda: float(Q("inf")) == float("inf"))
run("type(float(Q('inf')))", lambda: type(float(Q("inf"))).__name__)

print("[%s] string -> quad (special values + a couple normals):" % TAG)
for s in ["inf", "-inf", "nan", "-nan", "Infinity", "1.5", "0.1"]:
    run("Q(%r)" % s, lambda s=s: "float=%s bits=%s" % (fval(Q(s)), bits(Q(s))))

print("[%s] float() cast of constructed specials:" % TAG)
run("float(Q('inf'))", lambda: fval(Q("inf")))
run("np.isinf(float(Q('inf')))", lambda: np.isinf(fval(Q("inf"))))
run("np.isnan(float(Q('nan')))", lambda: np.isnan(fval(Q("nan"))))

print("[%s] frexp (quad mantissa/exp  vs  numpy float64):" % TAG)
for v in ["0.1", "0.9", "1.5", "inf", "nan"]:
    def _fx(v=v):
        q = Q(v)
        qm, qe = np.frexp(q)
        fm, fe = np.frexp(np.float64(fval(q)))
        return "quad=(%s, %s)  np=(%s, %s)" % (fval(qm), qe, fm, fe)
    run("frexp(Q(%s))" % v, _fx)

print("[%s] ldexp overflow (expect inf):" % TAG)
for base, e in [("1.5", 16384), ("2.0", 20000)]:
    run("ldexp(Q(%s), %d)" % (base, e),
        lambda base=base, e=e: fval(np.ldexp(Q(base), e)))

print("[%s] modf:" % TAG)
run("modf(Q('-0.001'))",
    lambda: tuple(fval(x) for x in np.modf(Q("-0.001"))))

print("[%s] matmul special (Windows uses the naive kernel; QBLAS disabled):" % TAG)
def _mm():
    A = np.array([[Q("inf"), Q("1")], [Q("2"), Q("3")]])
    Im = np.array([[Q("1"), Q("0")], [Q("0"), Q("1")]])
    return fval(np.matmul(A, Im)[0, 0])
run("matmul(inf-matrix, I)[0,0]", _mm)

print("########## end probe [%s] ##########\n" % TAG)
