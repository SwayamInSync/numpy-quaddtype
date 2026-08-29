"""Windows free-threaded isolation probe for numpy-quaddtype.

Prints the ACTUAL computed values for the operations that fail on the Windows
cp313t (free-threaded) wheel but pass on cp313 (GIL) and on Linux/macOS. Run the
SAME script under both interpreters and diff the ``[GIL]`` vs ``[FT]`` blocks to
see exactly what diverges.

This is a diagnostic, not a gate: it never raises to the shell (always exits 0),
so both wheels' output is captured in the CI log.
"""
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)  # keep subprocess output in order
except Exception:  # noqa: BLE001
    pass

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


def cpu_info():
    import platform
    lines = []
    try:
        import numpy._core._multiarray_umath as _mu
        lines.append("  numpy baseline      : %s" % (getattr(_mu, "__cpu_baseline__", "?"),))
        lines.append("  numpy dispatch      : %s" % (getattr(_mu, "__cpu_dispatch__", "?"),))
    except Exception as e:  # noqa: BLE001
        lines.append("  numpy cpu features  : <err %s>" % e)
    lines.append("  platform.processor  : %r" % platform.processor())
    lines.append("  platform.machine    : %r" % platform.machine())
    lines.append("  PROCESSOR_IDENTIFIER: %r" % os.environ.get("PROCESSOR_IDENTIFIER"))
    brand = None
    for cmd in (
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
        ["wmic", "cpu", "get", "name"],
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            txt = " / ".join(
                ln.strip() for ln in out.stdout.splitlines()
                if ln.strip() and ln.strip() != "Name"
            )
            if txt:
                brand = txt
                break
        except Exception:  # noqa: BLE001
            continue
    lines.append("  CPU brand           : %r" % (brand,))
    return "\n".join(lines)


TAG = "FT" if not gil_enabled() else "GIL"

print("\n########## quaddtype win-ft probe [%s] ##########" % TAG)
print("python    : %s  gil_enabled=%s" % (sys.version.split()[0], gil_enabled()))
print("numpy     : %s" % np.__version__)
print("quaddtype : %s" % getattr(nq, "__version__", "?"))
run("is_longdouble_128", nq.is_longdouble_128)
print("[%s] CPU / SIMD environment:" % TAG)
print(cpu_info())

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

# ---- BRANCH LOCALIZATION: same np.isinf, different input stride/shape ---------
# The CPU-dispatched DOUBLE_isinf loop branches on the INPUT stride:
#   0-D scalar   -> input stride 0        -> NCONTIG branch
#   1-D contig   -> stride == itemsize    -> CONTIG branch (this one works on FT-win)
#   1-D non-cont -> stride != itemsize, !0 -> NCONTIG branch (stride != 0)
# Comparing these tells us whether the fault is stride==0 only, any NCONTIG, or the
# small-count scalar tail in general.
print("[%s] BRANCH LOCALIZATION (isinf, varying input stride/shape):" % TAG)
run("0-D    isinf(np.array(np.inf))", lambda: bool(np.isinf(np.array(np.inf))))
run("1-Dx1  isinf([inf])[0]", lambda: bool(np.isinf(np.array([np.inf]))[0]))
run("1-Dx2  isinf([inf,inf])[0]", lambda: bool(np.isinf(np.array([np.inf, np.inf]))[0]))
run("1-Dx9  isinf([inf]*9).all()", lambda: bool(np.isinf(np.array([np.inf] * 9)).all()))
run("1-Dnc  isinf([inf,1][::2])[0]", lambda: bool(np.isinf(np.array([np.inf, 1.0])[::2])[0]))

# Same-dtype unary ufuncs go through the SCALAR FAST PATH (try_trivial_scalar_call),
# unlike isinf/isnan (bool output) which bail out of it. If these are ALSO wrong,
# the fast path is implicated; if they are fine, the fault is the normal 0-D path.
print("[%s] SAME-DTYPE fast-path scalar ufuncs (should be unaffected if 0-D path):" % TAG)
run("negative(np.inf)", lambda: float(np.negative(np.inf)))
run("negative(np.float64 inf)", lambda: float(np.negative(np.float64("inf"))))
run("absolute(np.float64 -inf)", lambda: float(np.absolute(np.float64("-inf"))))
run("sqrt(np.float64 inf)", lambda: float(np.sqrt(np.float64("inf"))))
run("signbit(np.float64 -inf)", lambda: bool(np.signbit(np.float64("-inf"))))

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

# ---- SIMD-DISABLED re-check: force the baseline (non-dispatched) loop -----------
# NPY_DISABLE_CPU_FEATURES is read at import, so re-run the key scalars in a fresh
# subprocess with every dispatched CPU feature disabled. If scalar isinf/isnan then
# become correct on FT-Windows, the fault is the CPU-dispatched SIMD DOUBLE_isinf
# codegen; if still wrong, it is the baseline/generic path.
print("[%s] SIMD-DISABLED re-check (baseline loop via NPY_DISABLE_CPU_FEATURES):" % TAG)
try:
    import numpy._core._multiarray_umath as _mu
    _feats = " ".join(getattr(_mu, "__cpu_dispatch__", []) or [])
except Exception as e:  # noqa: BLE001
    _feats = ""
    print("   could not read __cpu_dispatch__: %s" % e)
print("   dispatched features being disabled: %s" % (_feats or "(none)"))
_mini = (
    "import numpy as np;"
    "print('   [nosimd] numpy', np.__version__);"
    "print('   [nosimd] isinf(np.inf)         =', np.isinf(np.inf));"
    "print('   [nosimd] isnan(np.float64 nan) =', np.isnan(np.float64('nan')));"
    "print('   [nosimd] isinf([inf])[0]       =', np.isinf(np.array([np.inf]))[0])"
)
_env = dict(os.environ)
if _feats:
    _env["NPY_DISABLE_CPU_FEATURES"] = _feats
sys.stdout.flush()
try:
    subprocess.run([sys.executable, "-c", _mini], env=_env, check=False)
except Exception as e:  # noqa: BLE001
    print("   subprocess failed: %s" % e)

print("########## end probe [%s] ##########\n" % TAG)
