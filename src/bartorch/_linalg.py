"""LAPACK routines BART needs that the process does not already provide.

Each callback implements one Fortran-ABI LAPACK entry point on top of NumPy
(and SciPy where NumPy has no equivalent).  They are installed into the
compiled library's backend table for whichever routines no loaded library
exports; see :mod:`bartorch._backend`.

Every argument arrives as a raw pointer.  Matrices are column-major with a
leading dimension, exactly as LAPACK defines them, and are viewed in place.
A workspace query (``lwork == -1``) is answered with a size of one.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import traceback

import numpy as np

_log = logging.getLogger("bartorch.linalg")

_P = ctypes.c_void_p


def _fn(nargs: int):
    return ctypes.CFUNCTYPE(None, *([_P] * nargs))


def _int(p: int) -> int:
    return ctypes.c_int.from_address(p).value


def _set_int(p: int, v: int) -> None:
    ctypes.c_int.from_address(p).value = int(v)


def _char(p: int) -> str:
    return ctypes.c_char.from_address(p).value.decode().upper()


def _mat(p: int, rows: int, cols: int, ld: int, dtype) -> np.ndarray:
    """Column-major ``rows x cols`` view with leading dimension ``ld``."""
    dt = np.dtype(dtype)
    count = max(ld * (cols - 1) + rows, 0)
    if count == 0:
        return np.empty((rows, cols), dtype=dt, order="F")
    buf = (ctypes.c_char * (count * dt.itemsize)).from_address(p)
    flat = np.frombuffer(buf, dtype=dt, count=count)
    return np.lib.stride_tricks.as_strided(
        flat, shape=(rows, cols), strides=(dt.itemsize, ld * dt.itemsize), writeable=True
    )


def _vec(p: int, n: int, dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    if n == 0:
        return np.empty(0, dtype=dt)
    buf = (ctypes.c_char * (n * dt.itemsize)).from_address(p)
    return np.frombuffer(buf, dtype=dt, count=n)


def _query(work: int, lwork: int, dtype) -> bool:
    """Answer a workspace query; return True when the call was only a query."""
    if _int(lwork) == -1:
        _vec(work, 1, dtype)[0] = 1
        return True
    return False


def _guard(name, info):
    """Decorate a callback body so any failure reports through ``info``."""

    def wrap(body):
        try:
            _set_int(info, 0)
            body()
        except Exception:
            _log.error("%s failed:\n%s", name, traceback.format_exc())
            _set_int(info, -1)

    return wrap


def _scipy_linalg():
    try:
        import scipy.linalg as sl
    except ImportError as exc:
        raise RuntimeError("this LAPACK routine needs SciPy: pip install scipy") from exc
    return sl


# --- Hermitian eigenproblems ------------------------------------------------


def _heev(cdtype, rdtype):
    def cb(jobz, uplo, n, a, lda, w, work, lwork, rwork, info, *_):
        @_guard("heev", info)
        def _():
            if _query(work, lwork, cdtype):
                return
            nn = _int(n)
            A = _mat(a, nn, nn, _int(lda), cdtype)
            vals, vecs = np.linalg.eigh(A, UPLO=_char(uplo))
            _vec(w, nn, rdtype)[:] = vals
            if _char(jobz) == "V":
                A[:, :] = vecs

    return _fn(12)(cb)


def _chegv():
    def cb(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, info, *_):
        @_guard("chegv", info)
        def _():
            if _query(work, lwork, np.complex64):
                return
            sl = _scipy_linalg()
            nn = _int(n)
            A = _mat(a, nn, nn, _int(lda), np.complex64)
            B = _mat(b, nn, nn, _int(ldb), np.complex64)
            vals, vecs = sl.eigh(A, B, type=_int(itype), lower=(_char(uplo) == "L"))
            _vec(w, nn, np.float32)[:] = vals
            if _char(jobz) == "V":
                A[:, :] = vecs

    return _fn(15)(cb)


# --- Singular value decomposition -------------------------------------------


def _write_svd(A, s, u, ldu, vt, ldvt, jobu, jobvt, rdtype):
    m, n = A.shape
    k = min(m, n)
    compute_uv = (jobu != "N") or (jobvt != "N")
    full = (jobu == "A") or (jobvt == "A")
    if compute_uv:
        U, S, VT = np.linalg.svd(A, full_matrices=full)
    else:
        S = np.linalg.svd(A, compute_uv=False)
    _vec(s, k, rdtype)[:] = S
    if jobu in ("A", "S"):
        cols = m if jobu == "A" else k
        _mat(u, m, cols, _int(ldu), A.dtype)[:, :] = U[:, :cols]
    if jobvt in ("A", "S"):
        rows = n if jobvt == "A" else k
        _mat(vt, rows, n, _int(ldvt), A.dtype)[:, :] = VT[:rows, :]


def _gesdd(cdtype, rdtype):
    def cb(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info, *_):
        @_guard("gesdd", info)
        def _():
            if _query(work, lwork, cdtype):
                return
            job = _char(jobz)
            if job == "O":
                raise ValueError("gesdd jobz='O' is not supported by the NumPy backend")
            A = _mat(a, _int(m), _int(n), _int(lda), cdtype)
            _write_svd(A, s, u, ldu, vt, ldvt, job, job, rdtype)

    return _fn(16)(cb)


def _cgesvd():
    def cb(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info, *_):
        @_guard("cgesvd", info)
        def _():
            if _query(work, lwork, np.complex64):
                return
            ju, jv = _char(jobu), _char(jobvt)
            if "O" in (ju, jv):
                raise ValueError("gesvd job 'O' is not supported by the NumPy backend")
            A = _mat(a, _int(m), _int(n), _int(lda), np.complex64)
            _write_svd(A, s, u, ldu, vt, ldvt, ju, jv, np.float32)

    return _fn(17)(cb)


# --- QR ---------------------------------------------------------------------

# cgeqrf leaves R in the upper triangle and the reflectors below it; cungqr
# then forms Q in place.  NumPy hands back Q and R directly, so Q is parked
# between the two calls, keyed by the array it belongs to.
_pending_q: dict[int, np.ndarray] = {}


def _cgeqrf():
    def cb(m, n, a, lda, tau, work, lwork, info):
        @_guard("cgeqrf", info)
        def _():
            if _query(work, lwork, np.complex64):
                return
            mm, nn = _int(m), _int(n)
            A = _mat(a, mm, nn, _int(lda), np.complex64)
            Q, R = np.linalg.qr(A, mode="reduced")
            k = min(mm, nn)
            A[:k, :] = R
            _pending_q[a] = Q

    return _fn(8)(cb)


def _cungqr():
    def cb(m, n, k, a, lda, tau, work, lwork, info):
        @_guard("cungqr", info)
        def _():
            if _query(work, lwork, np.complex64):
                return
            Q = _pending_q.pop(a, None)
            if Q is None:
                raise RuntimeError("cungqr called without a preceding cgeqrf on the same array")
            mm, nn = _int(m), _int(n)
            A = _mat(a, mm, nn, _int(lda), np.complex64)
            A[:, :] = Q[:, :nn]

    return _fn(9)(cb)


# --- Cholesky and triangular systems ----------------------------------------


def _cpotrf():
    def cb(uplo, n, a, lda, info, *_):
        @_guard("cpotrf", info)
        def _():
            nn = _int(n)
            A = _mat(a, nn, nn, _int(lda), np.complex64)
            up = _char(uplo) == "U"
            H = np.triu(A) if up else np.tril(A)
            H = H + H.conj().T - np.diag(np.diag(H))
            try:
                L = np.linalg.cholesky(H)
            except np.linalg.LinAlgError:
                _set_int(info, 1)
                return
            if up:
                A[np.triu_indices(nn)] = L.conj().T[np.triu_indices(nn)]
            else:
                A[np.tril_indices(nn)] = L[np.tril_indices(nn)]

    return _fn(6)(cb)


def _triangle(A, uplo, diag):
    T = np.triu(A) if uplo == "U" else np.tril(A)
    if diag == "U":
        np.fill_diagonal(T, 1)
    return T


def _ctrtri():
    def cb(uplo, diag, n, a, lda, info, *_):
        @_guard("ctrtri", info)
        def _():
            nn = _int(n)
            A = _mat(a, nn, nn, _int(lda), np.complex64)
            up, dg = _char(uplo), _char(diag)
            T = _triangle(A, up, dg)
            Ti = np.linalg.inv(T)
            idx = np.triu_indices(nn) if up == "U" else np.tril_indices(nn)
            A[idx] = Ti[idx]

    return _fn(8)(cb)


def _ctrtrs():
    def cb(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, *_):
        @_guard("ctrtrs", info)
        def _():
            nn, nr = _int(n), _int(nrhs)
            A = _mat(a, nn, nn, _int(lda), np.complex64)
            B = _mat(b, nn, nr, _int(ldb), np.complex64)
            T = _triangle(A, _char(uplo), _char(diag))
            tr = _char(trans)
            if tr == "T":
                T = T.T
            elif tr == "C":
                T = T.conj().T
            B[:, :] = np.linalg.solve(T, B)

    return _fn(13)(cb)


# --- LU-based inverse and solve ---------------------------------------------


def _lu_reconstruct(LU, ipiv):
    n = LU.shape[0]
    L = np.tril(LU, -1) + np.eye(n, dtype=LU.dtype)
    U = np.triu(LU)
    A = L @ U
    for i in reversed(range(n)):
        p = ipiv[i] - 1
        if p != i:
            A[[i, p], :] = A[[p, i], :]
    return A


def _getrf(dtype):
    def cb(m, n, a, lda, ipiv, info):
        @_guard("getrf", info)
        def _():
            mm, nn = _int(m), _int(n)
            if mm != nn:
                raise ValueError("NumPy getrf supports square matrices only")
            sl = _scipy_linalg()
            A = _mat(a, mm, nn, _int(lda), dtype)
            lu, piv = sl.lu_factor(A)
            A[:, :] = lu
            _vec(ipiv, nn, np.int32)[:] = piv + 1

    return _fn(6)(cb)


def _getri(dtype):
    def cb(n, a, lda, ipiv, work, lwork, info):
        @_guard("getri", info)
        def _():
            if _query(work, lwork, dtype):
                return
            nn = _int(n)
            A = _mat(a, nn, nn, _int(lda), dtype)
            piv = _vec(ipiv, nn, np.int32)
            A[:, :] = np.linalg.inv(_lu_reconstruct(A.copy(), piv))

    return _fn(7)(cb)


def _sgesv():
    def cb(n, nrhs, a, lda, ipiv, b, ldb, info):
        @_guard("sgesv", info)
        def _():
            nn, nr = _int(n), _int(nrhs)
            A = _mat(a, nn, nn, _int(lda), np.float32)
            B = _mat(b, nn, nr, _int(ldb), np.float32)
            B[:, :] = np.linalg.solve(A, B)

    return _fn(8)(cb)


# --- Schur and Sylvester ----------------------------------------------------


def _gees(cdtype):
    def cb(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info, *_):
        @_guard("gees", info)
        def _():
            if _query(work, lwork, cdtype):
                return
            sl = _scipy_linalg()
            nn = _int(n)
            A = _mat(a, nn, nn, _int(lda), cdtype)
            T, Z = sl.schur(A, output="complex")
            A[:, :] = T
            _vec(w, nn, cdtype)[:] = np.diag(T)
            _set_int(sdim, 0)
            if _char(jobvs) == "V":
                _mat(vs, nn, nn, _int(ldvs), cdtype)[:, :] = Z

    return _fn(17)(cb)


def _ctrsyl():
    def cb(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info, *_):
        @_guard("ctrsyl", info)
        def _():
            sl = _scipy_linalg()
            mm, nn = _int(m), _int(n)
            A = _mat(a, mm, mm, _int(lda), np.complex64)
            B = _mat(b, nn, nn, _int(ldb), np.complex64)
            C = _mat(c, mm, nn, _int(ldc), np.complex64)
            ta, tb = _char(trana), _char(tranb)
            Aop = A.conj().T if ta == "C" else (A.T if ta == "T" else A)
            Bop = B.conj().T if tb == "C" else (B.T if tb == "T" else B)
            sign = _int(isgn)
            C[:, :] = sl.solve_sylvester(Aop, sign * Bop, C)
            ctypes.c_float.from_address(scale).value = 1.0

    return _fn(15)(cb)


def callbacks() -> dict[str, object]:
    """Fortran-ABI LAPACK callbacks keyed by symbol name."""
    return {
        "cheev_": _heev(np.complex64, np.float32),
        "zheev_": _heev(np.complex128, np.float64),
        "chegv_": _chegv(),
        "cgesdd_": _gesdd(np.complex64, np.float32),
        "zgesdd_": _gesdd(np.complex128, np.float64),
        "cgesvd_": _cgesvd(),
        "cgeqrf_": _cgeqrf(),
        "cungqr_": _cungqr(),
        "cpotrf_": _cpotrf(),
        "ctrtri_": _ctrtri(),
        "ctrtrs_": _ctrtrs(),
        "cgetrf_": _getrf(np.complex64),
        "sgetrf_": _getrf(np.float32),
        "cgetri_": _getri(np.complex64),
        "sgetri_": _getri(np.float32),
        "sgesv_": _sgesv(),
        "cgees_": _gees(np.complex64),
        "zgees_": _gees(np.complex128),
        "ctrsyl_": _ctrsyl(),
    }


if sys.platform == "win32":  # pragma: no cover
    _fn = lambda nargs: ctypes.CFUNCTYPE(None, *([_P] * nargs))  # noqa: E731
