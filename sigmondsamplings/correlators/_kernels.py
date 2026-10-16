"""Private NumPy kernels for the correlator calculators.

Pure NumPy on raw buffers with the sample axis last, ``(..., time, N)``: no SS
types, no xarray. The calculators in :mod:`.effective` wrap these, and a
compiled backend can later be dropped in behind the same interface.

Effective energies reproduce ``EffectiveEnergyCalculator`` in Sigmond's
``tasks/task_utils.cc`` element by element, including its ``rtsafe`` root
solve for the time-symmetric kinds: the same initial guess, bracketing walk,
tolerance and iteration limit, so a failure here is a failure there.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FORWARD",
    "FORWARD_CONST",
    "SYMMETRIC",
    "SYMMETRIC_CONST",
    "effective_energy",
]

# Sigmond's type codes.
FORWARD, SYMMETRIC, FORWARD_CONST, SYMMETRIC_CONST = 0, 1, 2, 3

_BRACKET_STEPS = 30
_XACC = 1e-10
_MAXIT = 200


def effective_energy(
    buf: np.ndarray,
    times: np.ndarray,
    *,
    kind: int,
    step: int,
    time_extent: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Effective energies of every element of a correlator buffer.

    Args:
        buf: Real correlator values, shape ``(..., n_time, N)``. The time axis
            is labeled by ``times``; exact neighbours are gathered by coordinate.
            Missing coordinates give NaN inputs.
        times: The integer time of each row, shape ``(n_time,)``.
        kind: One of :data:`FORWARD`, :data:`SYMMETRIC`, :data:`FORWARD_CONST`,
            :data:`SYMMETRIC_CONST`.
        step: The time step ``s``.
        time_extent: The lattice extent ``T``.

    Returns:
        ``(energy, failed)``, both shaped like ``buf``. ``energy[..., i, :]`` is
        keyed at the source time ``times[i]`` (no plot offset) and is NaN where
        ``failed``. An element fails where Sigmond's calculator does, where an
        input it needs is missing or non-finite, and where the result is not
        finite.
    """
    buf = np.asarray(buf, dtype=float)
    times = np.asarray(times)
    s, T = int(step), int(time_extent)
    positions = {int(t): i for i, t in enumerate(times)}

    def neighbour(offset):
        indices = np.array([positions.get(int(t) + offset, -1) for t in times], dtype=int)
        gathered = np.take(buf, np.maximum(indices, 0), axis=-2)
        return np.where((indices >= 0)[:, None], gathered, np.nan)

    c, fwd = buf, neighbour(s)
    t = times.reshape(-1, 1).astype(float)  # broadcasts against (..., time, N)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if kind == FORWARD:
            r = fwd / c
            ok = r > 0.0
            energy = -np.log(np.where(ok, r, 1.0)) / s
        elif kind == FORWARD_CONST:
            back = neighbour(-s)
            r = (fwd - c) / (c - back)
            ok = r > 0.0
            energy = -np.log(np.where(ok, r, 1.0)) / s
        elif kind == SYMMETRIC:
            near = t < T // 2
            valid = (t >= 0) & (t < T - s)
            # Past T/2, mirror onto t' = T - t - s with the inverted ratio.
            k = np.where(near, T - 2 * t, T - 2 * (T - t - s))
            r = np.where(near, fwd / c, c / fwd)
            ok = valid & (r >= 0.0) & (r < 1.0)
            energy, ok = _solve_symmetric(r, np.broadcast_to(k, r.shape), ok, s, kind)
        elif kind == SYMMETRIC_CONST:
            back = neighbour(-s)
            near = t < T // 2
            valid = (t >= 0) & (t < T - 2 * s)
            # Past T/2, mirror onto t' = T - t with the reflected differences.
            k = np.where(near, T - 2 * t, 2 * t - T)
            r = np.where(near, (fwd - c) / (c - back), (back - c) / (c - fwd))
            ok = valid & (r >= 0.0) & (r < 1.0)
            energy, ok = _solve_symmetric(r, np.broadcast_to(k, r.shape), ok, s, kind)
        else:
            raise ValueError(f"unknown effective-energy kind {kind!r}")

        ok &= np.isfinite(c) & np.isfinite(fwd) & np.isfinite(energy)
        if kind >= FORWARD_CONST:
            ok &= np.isfinite(back)
    energy = np.where(ok, energy, np.nan)
    return energy, ~ok


# ---------------------------------------------------------------------------
# Time-symmetric root solve
# ---------------------------------------------------------------------------


def _funcd(kind: int, b, r, k, s: int):
    """``f(b)`` and ``f'(b)``: Sigmond's ``PeriodicExpFuncDeriv`` / ``PeriodicExp2FuncDeriv``.

    Plain ``pow`` like Sigmond's, so results agree to rounding. Computing the
    powers as ``exp(k ln b)`` was measured and bought nothing: the cost is the
    handful of full-size Newton evaluations, not the power itself.
    """
    if kind == SYMMETRIC:
        # (1 + b^K) r - b^s - b^(K-s)
        b1, b2, b3 = np.power(b, k - 1), np.power(b, s - 1), np.power(b, k - s - 1)
        df = b1 * k * r - b2 * s - b3 * (k - s)
        f = (1.0 + b1 * b) * r - b * (b2 + b3)
    else:
        # (1 - b^(K+s)) r - b^s + b^K
        b1, b2 = np.power(b, k - 1), np.power(b, s - 1)
        df = -(k + s) * b1 * b2 * b * r - s * b2 + k * b1
        f = (1.0 - b1 * b2 * b * b) * r - b * (b2 - b1)
    return f, df


def _solve_symmetric(r, k, ok, s: int, kind: int):
    """Solve for ``b = e^{-m}`` at every element where ``ok``; returns ``(-ln b, ok)``."""
    energy = np.full(r.shape, np.nan)
    ok = ok.copy()
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return energy, ok
    root, solved = _bracket_and_solve(r.ravel()[idx], k.ravel()[idx], s, kind)
    flat_energy, flat_ok = energy.reshape(-1), ok.reshape(-1)
    flat_energy[idx] = np.where(solved, -np.log(np.where(solved, root, 1.0)), np.nan)
    flat_ok[idx] = solved
    return energy, ok


def _bracket_and_solve(r, k, s: int, kind: int):
    """Sigmond's bracketing walk then ``rtsafe``, on 1-D arrays of elements."""
    n = r.size
    failed = np.zeros(n, dtype=bool)

    # Initial guess, then walk the Newton step outward until the root is bracketed.
    sb = np.power(r, 1.0 / s)
    f, df = _funcd(kind, sb, r, k, s)
    bstep = -f / df
    sa = sb + 2.0 * bstep
    fnext, _ = _funcd(kind, sa, r, k, s)
    count = np.zeros(n, dtype=int)
    walking = f * fnext > 0
    while walking.any():
        w = np.flatnonzero(walking)
        sa[w] += bstep[w]
        bad = (sa[w] <= 0.0) | (sa[w] >= 1.0) | (count[w] >= _BRACKET_STEPS)
        failed[w[bad]] = True
        w = w[~bad]
        count[w] += 1
        fnext[w], _ = _funcd(kind, sa[w], r[w], k[w], s)
        walking[:] = False
        walking[w] = f[w] * fnext[w] > 0

    root = np.full(n, np.nan)
    solved = np.zeros(n, dtype=bool)
    live = np.flatnonzero(~failed)
    if live.size:
        root[live], solved[live] = _rtsafe(sa[live], sb[live], r[live], k[live], s, kind)
    return root, solved


def _rtsafe(x1, x2, r, k, s: int, kind: int):
    """Newton-Raphson with a bisection safeguard (Numerical Recipes ``rtsafe``).

    Each element runs the same sequence of steps as Sigmond's scalar version;
    only elements still iterating are evaluated.
    """
    n = x1.size
    root = np.full(n, np.nan)
    solved = np.zeros(n, dtype=bool)

    fl, _ = _funcd(kind, x1, r, k, s)
    fh, _ = _funcd(kind, x2, r, k, s)
    unbracketed = ((fl > 0.0) & (fh > 0.0)) | ((fl < 0.0) & (fh < 0.0))
    at_x1 = ~unbracketed & (fl == 0.0)
    at_x2 = ~unbracketed & ~at_x1 & (fh == 0.0)
    root[at_x1], solved[at_x1] = x1[at_x1], True
    root[at_x2], solved[at_x2] = x2[at_x2], True

    idx = np.flatnonzero(~(unbracketed | at_x1 | at_x2))
    x1, x2, r, k, fl = x1[idx], x2[idx], r[idx], k[idx], fl[idx]
    xl = np.where(fl < 0.0, x1, x2)
    xh = np.where(fl < 0.0, x2, x1)
    rts = 0.5 * (x1 + x2)
    dxold = np.abs(x2 - x1)
    dx = dxold.copy()
    f, df = _funcd(kind, rts, r, k, s)

    for _ in range(_MAXIT):
        if idx.size == 0:
            break
        bisect = (((rts - xh) * df - f) * ((rts - xl) * df - f) >= 0.0) | (
            np.abs(2.0 * f) > np.abs(dxold * df)
        )
        dxold = dx
        dx_bisect = 0.5 * (xh - xl)
        dx_newton = f / df
        new = np.where(bisect, xl + dx_bisect, rts - dx_newton)
        # Stop when the step no longer moves the estimate, or is below tolerance.
        stalled = np.where(bisect, xl == new, rts == new)
        dx = np.where(bisect, dx_bisect, dx_newton)
        rts = new
        done = stalled | (np.abs(dx) < _XACC)

        root[idx[done]], solved[idx[done]] = rts[done], True
        keep = ~done
        idx, rts, dx, dxold = idx[keep], rts[keep], dx[keep], dxold[keep]
        xl, xh, r, k = xl[keep], xh[keep], r[keep], k[keep]
        f, df = _funcd(kind, rts, r, k, s)
        below = f < 0.0
        xl = np.where(below, rts, xl)
        xh = np.where(below, xh, rts)
    # Anything still iterating hit the limit: Sigmond throws, which is a failure.
    return root, solved


# Single-pivot kernels. Matrix axes are last here; sampled callers move their
# sample axis before these axes explicitly.
def hermitian_upper(matrix, *, copy=True):
    """Copy Sigmond's upper triangle and real diagonal, without averaging."""
    matrix = np.asarray(matrix)
    result = matrix.copy() if copy else matrix
    i, j = np.triu_indices(matrix.shape[-1], 1)
    result[..., j, i] = result[..., i, j].conj()
    i = np.arange(matrix.shape[-1])
    result[..., i, i] = matrix[..., i, i].real
    return result


def pivot_cutoff(eigenvalues, epsilon, numeric_type):
    maximum = float(eigenvalues[-1])
    if numeric_type == "real" and maximum < epsilon:
        return epsilon
    return abs(epsilon * maximum)


def gevp_solve(A, B, epsilon, alarm, numeric_type):
    """Raw solve, with private ndarray results and no labeled dependencies."""
    b, P = np.linalg.eigh(B)
    if b[0] < alarm:
        raise ValueError(f"metric: eigenvalue {b[0]:g} below alarm {alarm:g}")
    bc = pivot_cutoff(b, epsilon, numeric_type)
    keep = b >= bc
    if not keep.any():
        raise ValueError("metric: all eigenvalues cut")
    if np.any(b[keep] <= 0):
        raise ValueError("metric: retained zero or negative eigenvalue cannot be inverted")
    W = P[:, keep] / np.sqrt(b[keep])
    G = hermitian_upper(W.conj().T @ A @ W)
    g, V = np.linalg.eigh(G)
    if g[0] < alarm:
        raise ValueError(f"matrix: eigenvalue {g[0]:g} below alarm {alarm:g}")
    gc = pivot_cutoff(g, epsilon, numeric_type)
    selected = np.flatnonzero(g >= gc)[::-1]
    if selected.size == 0:
        raise ValueError("matrix: all eigenvalues cut")
    eigenvalues = g[selected]
    Y = W @ V[:, selected]
    Z = B @ Y
    null = None
    if not keep.all():
        a, Q = np.linalg.eigh(A)
        ac = abs(epsilon * a[-1])
        D = P[:, ~keep]
        Qnull = Q[:, a < ac]
        projections = np.sum(abs(Qnull.conj().T @ D) ** 2, axis=0)
        elements = abs(np.einsum("ij,ij->j", D.conj(), A @ D))
        null = dict(matrix_eigenvalues=a, cutoff=ac, matrix_null_dimension=Qnull.shape[1],
                    metric_null_dimension=D.shape[1], projections=projections,
                    matrix_elements=elements, ok=bool(np.min(projections) > 0.9))
    return dict(eigenvalues=eigenvalues, eigenvectors=Y, zmatrix=Z,
                metric_eigenvalues=b, matrix_eigenvalues=g, metric_whitening=W,
                metric_rank=int(keep.sum()), rank=selected.size, metric_cutoff=bc,
                matrix_cutoff=gc, inverse_condition=float(eigenvalues[-1] / eigenvalues[0])
                if eigenvalues[0] != 0 else float("nan"), null_space=null,
                reconstruction_residual=float(np.linalg.norm(A - (Z * eigenvalues) @ Z.conj().T)
                                              / max(np.linalg.norm(A), np.finfo(float).tiny)))


def rotate_matrix(buffer, coefficients, *, canonical_hermitian=False):
    """``T† C T`` for ``(time, sink, source, sample)`` buffers.

    Batch matrix products over samples, with time chunks limiting temporary
    storage. The sample-last output stays contiguous for statistics and I/O.
    The same general kernel also supports non-Hermitian matrices.
    """
    dtype = np.result_type(buffer, coefficients)
    nt, ni, _, ns = buffer.shape
    no = coefficients.shape[1]
    result = np.empty((nt, no, no, ns), dtype=dtype)
    # Both products coexist briefly. Bound them to about 32 MiB, except when
    # a single time slice itself exceeds that budget.
    elements = no * (ni + no) + (ni * ni if canonical_hermitian else 0)
    bytes_per_time = max(1, ns * elements * dtype.itemsize)
    chunk = max(1, (32 * 1024**2) // bytes_per_time)
    left = coefficients.conj().T
    for start in range(0, nt, chunk):
        stop = min(nt, start + chunk)
        batched = np.moveaxis(buffer[start:stop], -1, 1)
        if canonical_hermitian:
            batched = hermitian_upper(batched)
        rotated = left @ batched @ coefficients
        if canonical_hermitian:
            hermitian_upper(rotated, copy=False)
        result[start:stop] = np.moveaxis(rotated, 1, -1)
        del batched, rotated
    return result
