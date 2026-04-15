"""DGA-based committor estimation for path sampling (k-means basis on VAC coords).

We **prefer** to use `extq` when it is importable (it provides a reference DGA
implementation), but `extq` depends on `numba` and may not be available in all
environments. When `extq` is not importable, we fall back to a lightweight
NumPy/SciPy implementation (no `numba` dependency), sufficient for forward
committor estimation with stopping times (absorbing A/B boundaries).

We use a sparse indicator-function basis constructed from k-means microstates
in VAC (linear) coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from contextlib import contextmanager
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


_EXTQ_CACHE: Optional[tuple[Any, Any]] = None  # (extq.dga, extq.stop)
_EXTQ_TRIED = False
_EXTQ_DISABLED: bool = bool(int(os.environ.get("GESPO_DISABLE_EXTQ", "0") or "0"))
_EXTQ_DISABLED_REASON: str = "env:GESPO_DISABLE_EXTQ" if _EXTQ_DISABLED else ""


def _disable_extq_runtime(reason: str) -> None:
    """Disable extq for the remainder of this process after a runtime failure."""
    global _EXTQ_DISABLED, _EXTQ_DISABLED_REASON
    if _EXTQ_DISABLED:
        return
    _EXTQ_DISABLED = True
    _EXTQ_DISABLED_REASON = str(reason or "runtime")
    print(
        "[Committor][DGA] Note: disabling extq for the remainder of this process "
        "(set GESPO_DISABLE_EXTQ=1 or --committor-dga-backend=numpy to silence; "
        "use --committor-dga-backend=extq to force retries).",
        flush=True,
    )


@dataclass(frozen=True)
class _SolveInfo:
    method: str
    ridge_eps: float = 0.0
    note: str = ""


@contextmanager
def _silence_output_fds(*fds: int):
    """Silence low-level FD writes (e.g., SuperLU 'dgstrf info' messages)."""
    devnull_fd: Optional[int] = None
    saved: List[int] = []
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        for fd in fds:
            try:
                saved.append(os.dup(int(fd)))
                os.dup2(int(devnull_fd), int(fd))
            except Exception:
                saved.append(-1)
        yield
    finally:
        for fd, dup_fd in zip(fds, saved):
            try:
                if int(dup_fd) >= 0:
                    os.dup2(int(dup_fd), int(fd))
            except Exception:
                pass
            try:
                if int(dup_fd) >= 0:
                    os.close(int(dup_fd))
            except Exception:
                pass
        try:
            if devnull_fd is not None:
                os.close(int(devnull_fd))
        except Exception:
            pass


def _sanitize_weights(weights: Sequence[np.ndarray], *, basis: Sequence[scipy.sparse.spmatrix], lag: int) -> List[np.ndarray]:
    """Validate/clean DGA weights.

    extq occasionally returns non-finite/negative weights under ill-conditioned
    bases (e.g., rare/disconnected clusters early in a run). We treat such cases
    conservatively and fall back to uniform weights (with the required trailing
    zeros) rather than propagating NaNs into the committor solve.
    """

    lag_i = int(lag)
    if lag_i <= 0:
        lag_i = 1

    out: List[np.ndarray] = []
    for w, b in zip(weights, basis):
        n = int(getattr(b, "shape", (0,))[0] or 0)
        w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
        ok = True
        if int(w_arr.size) != n or n <= lag_i:
            ok = False
        if ok:
            w_arr = np.where(np.isfinite(w_arr), w_arr, 0.0)
            w_arr = np.clip(w_arr, 0.0, None)
            # DGA requires weights be zero in the last lag frames.
            w_arr[-lag_i:] = 0.0
            if not np.any(w_arr[:-lag_i] > 0.0):
                ok = False
        if not ok:
            w_arr = np.ones((max(0, n),), dtype=np.float64)
            if n >= lag_i:
                w_arr[-lag_i:] = 0.0
        out.append(w_arr.astype(np.float64, copy=False))
    return out


def _delay_embed_single(traj: np.ndarray, *, n_embed: int, lag: int) -> np.ndarray:
    """Delay-embed a single trajectory (notebook-style; see adk/bin/util.py:delay_embed).

    Parameters
    ----------
    traj
        Array of shape (T, D).
    n_embed
        Number of additional embedded snapshots (0 disables).
    lag
        Spacing between embedded snapshots (in trajectory frames).

    Returns
    -------
    np.ndarray
        Embedded array of shape (T - lag*n_embed, D*(n_embed+1)).
    """

    n_embed_i = int(n_embed)
    if n_embed_i <= 0:
        return np.asarray(traj, dtype=np.float64)

    lag_i = int(lag)
    if lag_i <= 0:
        lag_i = 1

    z = np.asarray(traj, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("traj must have shape (T, D)")

    T = int(z.shape[0])
    total = int(lag_i) * int(n_embed_i)
    out_T = int(T - total)
    if out_T <= 0:
        return np.zeros((0, int(z.shape[1]) * int(n_embed_i + 1)), dtype=np.float64)

    blocks: List[np.ndarray] = []
    for n in range(int(n_embed_i) + 1):
        start = int(lag_i) * int(n_embed_i - n)
        stop = T - int(lag_i) * int(n)
        blocks.append(z[start:stop])
    return np.concatenate(blocks, axis=1)


def _lift_center_single(fxn: np.ndarray, *, n_embed: int, lag: int, target_len: int) -> np.ndarray:
    """Lift a per-frame function to the center of the delay-embedding window.

    This mirrors the notebook's intent (center-of-window assignment) and produces
    a vector aligned to the embedded time axis.
    """

    n_embed_i = int(n_embed)
    if n_embed_i <= 0:
        return np.asarray(fxn).reshape(-1)

    lag_i = int(lag)
    if lag_i <= 0:
        lag_i = 1

    fxn_arr = np.asarray(fxn).reshape(-1)
    T = int(fxn_arr.shape[0])
    total = int(lag_i) * int(n_embed_i)
    out_T = int(T - total)
    if out_T <= 0:
        return np.zeros((0,), dtype=fxn_arr.dtype)

    pad_left = int(total // 2)
    pad_right = int(total - pad_left)
    lifted = fxn_arr[pad_left : T - pad_right]
    if int(target_len) > 0 and int(lifted.shape[0]) != int(target_len):
        m = int(min(int(lifted.shape[0]), int(target_len)))
        lifted = lifted[:m]
    return lifted


def _get_extq(*, force: bool = False) -> Optional[tuple[Any, Any]]:
    """Best-effort import of extq (cached). Returns (extq.dga, extq.stop) or None."""
    global _EXTQ_CACHE, _EXTQ_TRIED, _EXTQ_DISABLED
    if _EXTQ_DISABLED and not force:
        return None
    if _EXTQ_TRIED:
        return _EXTQ_CACHE
    _EXTQ_TRIED = True
    try:
        import extq.dga as extq_dga  # type: ignore
        import extq.stop as extq_stop  # type: ignore

        _EXTQ_CACHE = (extq_dga, extq_stop)
    except Exception:
        _EXTQ_CACHE = None
    return _EXTQ_CACHE


def forward_stop(in_domain: np.ndarray) -> np.ndarray:
    """First exit time from the domain (prefers extq.stop.forward_stop when available)."""
    extq = _get_extq()
    if extq is not None:
        _, extq_stop = extq
        fn = getattr(extq_stop, "forward_stop", None)
        if callable(fn):
            try:
                out = fn(in_domain)
                return np.asarray(out, dtype=np.int64).reshape(-1)
            except Exception:
                pass

    in_domain = np.asarray(in_domain, dtype=bool).reshape(-1)
    n = int(in_domain.shape[0])
    (t,) = np.nonzero(~in_domain)
    t = np.concatenate([[-1], t, [n]])
    out = np.repeat(t[1:], np.diff(t))[:-1]
    return out.astype(np.int64, copy=False)


def _solve(a, b, *, return_info: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, _SolveInfo]]:
    """
    Solve a linear system A x = b.

    In practice, DGA normal equations can be rank-deficient when the k-means basis
    contains disconnected/rare clusters (common early in a run with limited history).
    We therefore fall back to a small ridge regularization and least-squares solvers
    when the direct sparse solve reports singularity.
    """

    def _pack(x: np.ndarray, info: _SolveInfo) -> Union[np.ndarray, Tuple[np.ndarray, _SolveInfo]]:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return (x_arr, info) if return_info else x_arr

    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)

    # SciPy exposes MatrixRankWarning for sparse solves, but guard for API differences.
    try:
        MatrixRankWarning = scipy.sparse.linalg.MatrixRankWarning  # type: ignore[attr-defined]
    except Exception:
        MatrixRankWarning = UserWarning  # type: ignore[assignment]

    if scipy.sparse.issparse(a):
        a_csr = a.tocsr()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=MatrixRankWarning)
                with _silence_output_fds(1, 2):
                    x0 = scipy.sparse.linalg.spsolve(a_csr, b_arr)
                return _pack(x0, _SolveInfo(method="spsolve"))
        except Exception as exc:
            # Regularize the diagonal and retry. Use a scale-aware epsilon so that it is
            # meaningful across problems.
            try:
                diag = np.asarray(a_csr.diagonal(), dtype=np.float64)
                scale = float(np.median(np.abs(diag[np.isfinite(diag)]))) if diag.size else 1.0
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
            except Exception:
                scale = 1.0

            eps = max(1e-10, 1e-8 * float(scale))
            try:
                a_reg = a_csr + scipy.sparse.eye(a_csr.shape[0], format="csr") * eps
            except Exception:
                a_reg = a_csr

            # Retry sparse direct solve, then fall back to least-squares (LSMR).
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=MatrixRankWarning)
                    with _silence_output_fds(1, 2):
                        x = scipy.sparse.linalg.spsolve(a_reg, b_arr)
                    x = np.asarray(x, dtype=np.float64).reshape(-1)
                    if x.size == b_arr.size and np.all(np.isfinite(x)):
                        print(
                            f"[Committor][DGA] Warning: singular/unstable sparse solve ({exc}); "
                            f"used ridge regularization eps={eps:.2e}.",
                            flush=True,
                        )
                        return _pack(x, _SolveInfo(method="ridge_spsolve", ridge_eps=float(eps), note=str(exc)))
            except Exception:
                pass

            try:
                x = scipy.sparse.linalg.lsmr(a_reg, b_arr, damp=eps)[0]
                x = np.asarray(x, dtype=np.float64).reshape(-1)
                print(
                    f"[Committor][DGA] Warning: singular/unstable sparse solve ({exc}); "
                    f"used regularized least-squares (lsmr) eps={eps:.2e}.",
                    flush=True,
                )
                return _pack(x, _SolveInfo(method="lsmr", ridge_eps=float(eps), note=str(exc)))
            except Exception:
                # Last-resort: dense least-squares (can be expensive, but small K is typical).
                a_dense = np.asarray(a_reg.todense(), dtype=np.float64)
                x, *_ = np.linalg.lstsq(a_dense, b_arr, rcond=None)
                print(
                    f"[Committor][DGA] Warning: singular/unstable sparse solve ({exc}); "
                    "used dense least-squares fallback.",
                    flush=True,
                )
                return _pack(np.asarray(x, dtype=np.float64).reshape(-1), _SolveInfo(method="dense_lstsq", ridge_eps=float(eps), note=str(exc)))

    a_dense = np.asarray(a, dtype=np.float64)
    try:
        x0 = scipy.linalg.solve(a_dense, b_arr)
        return _pack(x0, _SolveInfo(method="dense_solve"))
    except Exception as exc:
        x, *_ = np.linalg.lstsq(a_dense, b_arr, rcond=None)
        print(
            f"[Committor][DGA] Warning: dense solve failed ({exc}); used least-squares fallback.",
            flush=True,
        )
        return _pack(np.asarray(x, dtype=np.float64).reshape(-1), _SolveInfo(method="dense_lstsq", note=str(exc)))


def dga_reweight(
    basis: Sequence[scipy.sparse.spmatrix],
    *,
    lag: int,
    maxlag: Optional[int] = None,
    guess: Optional[Sequence[np.ndarray]] = None,
    test_basis: Optional[Sequence[scipy.sparse.spmatrix]] = None,
) -> List[np.ndarray]:
    """Estimate change of measure to the invariant distribution (extq.dga.reweight, NumPy/SciPy port)."""
    lag_i = int(lag)
    if lag_i <= 0:
        raise ValueError("lag must be > 0")
    maxlag_i = lag_i if maxlag is None else int(maxlag)
    if maxlag_i < lag_i:
        raise ValueError("maxlag must be >= lag")

    if test_basis is None:
        test_basis = basis

    if guess is None:
        guess_list: List[np.ndarray] = []
        for x in basis:
            w = np.ones(int(x.shape[0]), dtype=np.float64)
            w[-maxlag_i:] = 0.0
            guess_list.append(w)
        guess = guess_list

    a = 0.0
    b = 0.0
    for x, y, w in zip(test_basis, basis, guess):
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if int(w.shape[0]) != int(x.shape[0]):
            raise ValueError("Weight length mismatch.")
        if not np.all(w[-maxlag_i:] == 0.0):
            raise ValueError("Weights must be zero in the last maxlag frames.")
        wdx = scipy.sparse.diags(w[:-lag_i], format="csr") @ (x[lag_i:] - x[:-lag_i])
        a += wdx.T @ y[:-lag_i]
        b -= np.ravel(wdx.sum(axis=0))

    coeffs = _solve(a, b)  # type: ignore[assignment]
    out: List[np.ndarray] = []
    for y, w in zip(basis, guess):
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        out.append((w * (y @ coeffs + 1.0)).astype(np.float64, copy=False))
    return out


def dga_forward_committor_coeffs(
    basis: Sequence[scipy.sparse.spmatrix],
    weights: Sequence[np.ndarray],
    in_domain: Sequence[np.ndarray],
    guess: Sequence[np.ndarray],
    *,
    lag: int,
    test_basis: Optional[Sequence[scipy.sparse.spmatrix]] = None,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, _SolveInfo]]:
    """Compute DGA forward-committor coefficients (extq.dga.forward_committor internal solve)."""
    lag_i = int(lag)
    if lag_i <= 0:
        raise ValueError("lag must be > 0")
    if test_basis is None:
        test_basis = basis

    a = 0.0
    b = 0.0
    for x, y, w, d, g in zip(test_basis, basis, weights, in_domain, guess):
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        d = np.asarray(d, dtype=bool).reshape(-1)
        g = np.asarray(g, dtype=np.float64).reshape(-1)

        n = int(d.shape[0])
        if int(x.shape[0]) != n or int(y.shape[0]) != n or int(w.shape[0]) != n or int(g.shape[0]) != n:
            raise ValueError("Per-trajectory length mismatch in DGA inputs.")
        if n <= lag_i:
            continue
        if not np.all(w[-lag_i:] == 0.0):
            raise ValueError("DGA weights must be zero in the last lag frames.")

        exit_time = forward_stop(d)
        iy = np.minimum(np.arange(lag_i, n, dtype=np.int64), exit_time[:-lag_i])
        wx = scipy.sparse.diags(w[:-lag_i], format="csr") @ x[:-lag_i]
        a += wx.T @ (y[iy] - y[:-lag_i])
        b -= wx.T @ (g[iy] - g[:-lag_i])

    if return_info:
        return _solve(a, b, return_info=True)
    return _solve(a, b)  # type: ignore[return-value]


@dataclass
class DGAKMeansCommittorModel:
    """Forward committor model q^+(x) trained via DGA on k-means VAC microstates."""

    kmeans: MiniBatchKMeans
    nn_a: Optional[NearestNeighbors]
    nn_b: Optional[NearestNeighbors]
    coeffs: np.ndarray  # shape (K,)
    basis_scale: np.ndarray  # shape (K,)
    dim: int
    lag: int
    delay_n_embed: int = 0
    delay_delta: int = 1
    eps: float = 1e-8
    backend: str = "numpy"
    # Optional feature normalization (z -> (z - mu)/std) used for k-means + NN guess.
    z_mu: Optional[np.ndarray] = None  # shape (dim,)
    z_std: Optional[np.ndarray] = None  # shape (dim,)
    # Diagnostics about the last fit (used to decide whether to trust q for rewards).
    solve_method: str = ""
    solve_ridge_eps: float = 0.0
    degenerate: bool = False
    q_domain_std: float = float("nan")

    def _standardize(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=np.float64)
        mu = self.z_mu
        std = self.z_std
        if mu is None or std is None:
            return z_arr
        mu_arr = np.asarray(mu, dtype=np.float64).reshape(1, -1)
        std_arr = np.asarray(std, dtype=np.float64).reshape(1, -1)
        if mu_arr.shape[1] != z_arr.shape[1] or std_arr.shape[1] != z_arr.shape[1]:
            # Shape mismatch: fall back to raw features rather than silently broadcasting.
            return z_arr
        std_arr = np.where(std_arr > 1e-8, std_arr, 1.0)
        return (z_arr - mu_arr) / std_arr

    def _guess(self, z_std: np.ndarray, in_a: np.ndarray, in_b: np.ndarray) -> np.ndarray:
        in_a = np.asarray(in_a, dtype=bool).reshape(-1)
        in_b = np.asarray(in_b, dtype=bool).reshape(-1)
        if self.nn_a is None or self.nn_b is None:
            guess = in_b.astype(np.float64, copy=False)
            guess[in_a] = 0.0
            guess[in_b] = 1.0
            return guess.astype(np.float64, copy=False)

        z_use = np.asarray(z_std, dtype=np.float64)
        if z_use.ndim != 2:
            raise ValueError("z must have shape (T, D)")
        if int(z_use.shape[1]) != int(self.dim):
            z_use = z_use[:, : int(self.dim)]

        da = self.nn_a.kneighbors(z_use, n_neighbors=1, return_distance=True)[0].reshape(-1)
        db = self.nn_b.kneighbors(z_use, n_neighbors=1, return_distance=True)[0].reshape(-1)
        da2 = da * da
        db2 = db * db
        denom = da2 + db2 + float(self.eps)
        guess = da2 / denom
        guess = np.where(np.isfinite(guess), guess, 0.0)
        guess = np.clip(guess, 0.0, 1.0)
        guess[in_a] = 0.0
        guess[in_b] = 1.0
        return guess.astype(np.float64, copy=False)

    def predict_q(self, z: np.ndarray, *, in_a: np.ndarray, in_b: np.ndarray) -> np.ndarray:
        q, _labels = self.predict_q_and_labels(z, in_a=in_a, in_b=in_b)
        return q

    def predict_q_and_labels(self, z: np.ndarray, *, in_a: np.ndarray, in_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_raw = np.asarray(z, dtype=np.float64)
        in_a_raw = np.asarray(in_a, dtype=bool).reshape(-1)
        in_b_raw = np.asarray(in_b, dtype=bool).reshape(-1)
        if z_raw.ndim != 2:
            raise ValueError("z must have shape (T, D)")
        if int(z_raw.shape[0]) != int(in_a_raw.shape[0]) or int(in_a_raw.shape[0]) != int(in_b_raw.shape[0]):
            raise ValueError("Length mismatch between z and boundary masks.")

        z_use_full = z_raw
        in_a_use = in_a_raw
        in_b_use = in_b_raw

        # Optional notebook-style delay embedding on z, with boundary masks lifted to
        # the center of the embedding window.
        if int(self.delay_n_embed) > 0:
            z_emb = _delay_embed_single(z_raw, n_embed=int(self.delay_n_embed), lag=int(self.delay_delta))
            T_emb = int(z_emb.shape[0])
            in_a_lift = _lift_center_single(
                in_a_raw,
                n_embed=int(self.delay_n_embed),
                lag=int(self.delay_delta),
                target_len=T_emb,
            ).astype(bool, copy=False)
            in_b_lift = _lift_center_single(
                in_b_raw,
                n_embed=int(self.delay_n_embed),
                lag=int(self.delay_delta),
                target_len=T_emb,
            ).astype(bool, copy=False)
            m = int(min(int(T_emb), int(in_a_lift.shape[0]), int(in_b_lift.shape[0])))
            z_use_full = z_emb[:m]
            in_a_use = in_a_lift[:m]
            in_b_use = in_b_lift[:m]

        T = int(z_use_full.shape[0])
        if T == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        z_use = z_use_full[:, : int(self.dim)]
        z_std = self._standardize(np.asarray(z_use, dtype=np.float64))
        labels = self.kmeans.predict(np.asarray(z_std, dtype=np.float64))
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        K = int(self.coeffs.shape[0])

        in_domain = ~(in_a_use | in_b_use)
        rows = np.nonzero(in_domain)[0].astype(np.int64)
        cols = labels[rows]
        data = np.ones(rows.shape[0], dtype=np.float64)
        basis = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(T, K))
        # Apply the orthogonalization scaling used during fitting.
        basis_d = basis @ scipy.sparse.diags(self.basis_scale, format="csr")

        guess = self._guess(z_std, in_a=in_a_use, in_b=in_b_use)
        q = np.asarray(basis_d @ self.coeffs, dtype=np.float64).reshape(-1) + guess
        q = np.where(np.isfinite(q), q, guess)
        q = np.clip(q, 0.0, 1.0)
        q[in_a_use] = 0.0
        q[in_b_use] = 1.0
        return q.astype(np.float32, copy=False), labels


def fit_dga_committor_kmeans(
    z_trajs: Sequence[np.ndarray],
    *,
    in_a_trajs: Sequence[np.ndarray],
    in_b_trajs: Sequence[np.ndarray],
    lag: int,
    n_clusters: int,
    dim: int,
    max_fit_samples: int,
    seed: int,
    backend: str = "auto",
    delay_n_embed: int = 0,
    delay_delta: int = 1,
    max_boundary_points: int = 5000,
    eps: float = 1e-8,
    fixed_kmeans: Optional[MiniBatchKMeans] = None,
    fixed_z_mu: Optional[np.ndarray] = None,
    fixed_z_std: Optional[np.ndarray] = None,
) -> Optional[DGAKMeansCommittorModel]:
    """Fit a DGA forward-committor model on VAC coordinates using a k-means indicator basis."""
    if len(z_trajs) != len(in_a_trajs) or len(z_trajs) != len(in_b_trajs):
        raise ValueError("z_trajs, in_a_trajs, and in_b_trajs must have the same length.")
    if not z_trajs:
        return None

    lag_i = int(lag)
    if lag_i <= 0:
        raise ValueError("lag must be > 0")
    K = int(n_clusters)
    if K <= 1:
        raise ValueError("n_clusters must be > 1")

    use_fixed_kmeans = fixed_kmeans is not None
    if use_fixed_kmeans:
        K_fixed = 0
        try:
            K_fixed = int(getattr(fixed_kmeans, "n_clusters", 0) or 0)
        except Exception:
            K_fixed = 0
        if K_fixed <= 0:
            try:
                K_fixed = int(np.asarray(getattr(fixed_kmeans, "cluster_centers_", np.zeros((0,)))).shape[0])
            except Exception:
                K_fixed = 0
        if K_fixed <= 1:
            raise ValueError("fixed_kmeans must have n_clusters > 1")
        if int(K_fixed) != int(K):
            print(
                f"[Committor][DGA] Note: overriding requested K={int(K)} with fixed_kmeans K={int(K_fixed)}.",
                flush=True,
            )
        K = int(K_fixed)

    delay_n = int(delay_n_embed)
    if delay_n < 0:
        delay_n = 0
    delay_delta_i = int(delay_delta)
    if delay_delta_i <= 0:
        delay_delta_i = 1
    if use_fixed_kmeans and delay_n > 0:
        raise ValueError("fixed_kmeans is incompatible with delay embedding (delay_n_embed>0).")

    # Optional notebook-style delay embedding: embed z(t) and lift boundary masks to
    # the center of the embedding window.
    z_trajs_use: List[np.ndarray] = []
    in_a_trajs_use: List[np.ndarray] = []
    in_b_trajs_use: List[np.ndarray] = []
    if delay_n > 0:
        for z, in_a, in_b in zip(z_trajs, in_a_trajs, in_b_trajs):
            try:
                z_arr = np.asarray(z, dtype=np.float64)
                ia = np.asarray(in_a, dtype=bool).reshape(-1)
                ib = np.asarray(in_b, dtype=bool).reshape(-1)
            except Exception:
                continue
            if z_arr.ndim != 2:
                continue
            T_raw = int(z_arr.shape[0])
            if int(ia.shape[0]) != T_raw or int(ib.shape[0]) != T_raw:
                continue
            z_emb = _delay_embed_single(z_arr, n_embed=int(delay_n), lag=int(delay_delta_i))
            T_emb = int(z_emb.shape[0])
            if T_emb <= int(lag_i):
                continue
            ia_lift = _lift_center_single(ia, n_embed=int(delay_n), lag=int(delay_delta_i), target_len=T_emb).astype(bool, copy=False)
            ib_lift = _lift_center_single(ib, n_embed=int(delay_n), lag=int(delay_delta_i), target_len=T_emb).astype(bool, copy=False)
            m = int(min(int(T_emb), int(ia_lift.shape[0]), int(ib_lift.shape[0])))
            if m <= int(lag_i):
                continue
            z_trajs_use.append(np.asarray(z_emb[:m], dtype=np.float64))
            in_a_trajs_use.append(np.asarray(ia_lift[:m], dtype=bool))
            in_b_trajs_use.append(np.asarray(ib_lift[:m], dtype=bool))
    else:
        for z, in_a, in_b in zip(z_trajs, in_a_trajs, in_b_trajs):
            try:
                z_trajs_use.append(np.asarray(z, dtype=np.float64))
                in_a_trajs_use.append(np.asarray(in_a, dtype=bool).reshape(-1))
                in_b_trajs_use.append(np.asarray(in_b, dtype=bool).reshape(-1))
            except Exception:
                continue

    if not z_trajs_use:
        return None

    # Determine feature dimension.
    z0 = np.asarray(z_trajs_use[0])
    if z0.ndim != 2:
        raise ValueError("Each z_traj must have shape (T, D).")
    d_total = int(z0.shape[1])
    dim_i = int(dim)
    if dim_i <= 0:
        dim_i = int(d_total) if delay_n > 0 else int(min(d_total, 5))
    dim_i = int(max(1, min(dim_i, d_total)))

    mu_fixed: Optional[np.ndarray] = None
    std_fixed: Optional[np.ndarray] = None
    if use_fixed_kmeans:
        if fixed_z_mu is None or fixed_z_std is None:
            raise ValueError("fixed_kmeans requires fixed_z_mu and fixed_z_std.")
        mu_fixed = np.asarray(fixed_z_mu, dtype=np.float64).reshape(-1)
        std_fixed = np.asarray(fixed_z_std, dtype=np.float64).reshape(-1)
        dim_fixed = int(mu_fixed.shape[0])
        if int(std_fixed.shape[0]) != int(dim_fixed):
            raise ValueError("fixed_z_mu and fixed_z_std must have the same length.")
        if int(dim) <= 0:
            dim_i = int(max(1, min(dim_fixed, d_total)))
        elif int(dim_i) != int(dim_fixed):
            raise ValueError(f"Requested dim={int(dim_i)} does not match fixed dim={int(dim_fixed)}.")
        if int(dim_i) != int(dim_fixed):
            raise ValueError(f"Fixed dim={int(dim_fixed)} is incompatible with inferred dim={int(dim_i)}.")

    rng = np.random.default_rng(int(seed))

    # Collect domain frames for k-means fitting.
    samples: List[np.ndarray] = []
    total = 0
    for z, in_a, in_b in zip(z_trajs_use, in_a_trajs_use, in_b_trajs_use):
        z = np.asarray(z, dtype=np.float64)
        in_domain = ~(np.asarray(in_a, dtype=bool).reshape(-1) | np.asarray(in_b, dtype=bool).reshape(-1))
        if z.ndim != 2 or int(z.shape[0]) != int(in_domain.shape[0]):
            continue
        z_use = z[:, :dim_i]
        idx = np.nonzero(in_domain)[0]
        if idx.size == 0:
            continue
        if max_fit_samples > 0 and total + int(idx.size) > int(max_fit_samples):
            need = max(0, int(max_fit_samples) - total)
            if need <= 0:
                break
            pick = rng.choice(idx, size=int(need), replace=False)
            idx = np.asarray(pick, dtype=np.int64)
        samples.append(np.asarray(z_use[idx], dtype=np.float64))
        total += int(idx.size)
        if max_fit_samples > 0 and total >= int(max_fit_samples):
            break

    if not samples:
        return None
    fit_arr = np.concatenate(samples, axis=0)
    if fit_arr.shape[0] < 2:
        return None

    # Normalize VAC coordinates before k-means and nearest-neighbor guesses.
    # VAC dims can have very different scales early in training; without normalization,
    # distance-based guesses can collapse to ~0/1 and k-means becomes brittle.
    if use_fixed_kmeans:
        assert mu_fixed is not None and std_fixed is not None
        mu = np.asarray(mu_fixed, dtype=np.float64).reshape(-1)
        std = np.asarray(std_fixed, dtype=np.float64).reshape(-1)
    else:
        mu = np.asarray(fit_arr.mean(axis=0), dtype=np.float64).reshape(-1)
        std = np.asarray(fit_arr.std(axis=0), dtype=np.float64).reshape(-1)
    std = np.where(std > 1e-6, std, 1.0)
    fit_arr = (fit_arr - mu[None, :]) / std[None, :]

    # Guardrail: avoid requesting more clusters than the available sample budget can support.
    # Too-large K leads to empty/rare clusters and ill-conditioned DGA solves (extq is
    # especially sensitive). Use a conservative minimum average occupancy.
    min_points_per_cluster = 20
    if int(fit_arr.shape[0]) < int(K) * int(min_points_per_cluster):
        K_suggest = max(2, int(fit_arr.shape[0]) // int(min_points_per_cluster))
        if K_suggest < int(K) and (not use_fixed_kmeans):
            print(
                f"[Committor][DGA] Warning: reducing k-means clusters from K={int(K)} to K={int(K_suggest)} "
                f"(fit_samples={int(fit_arr.shape[0])}, min_points_per_cluster={int(min_points_per_cluster)}).",
                flush=True,
            )
            K = int(K_suggest)
        elif use_fixed_kmeans and K_suggest < int(K):
            print(
                f"[Committor][DGA] Warning: fixed k-means basis uses K={int(K)} but fit_samples={int(fit_arr.shape[0])} "
                f"< K*{int(min_points_per_cluster)}; expect instability. Consider reducing K.",
                flush=True,
            )
    if (not use_fixed_kmeans) and int(K) > int(fit_arr.shape[0]):
        K = int(max(2, min(int(K), int(fit_arr.shape[0]))))
    elif use_fixed_kmeans and int(K) > int(fit_arr.shape[0]):
        print(
            f"[Committor][DGA] Warning: fixed k-means basis uses K={int(K)} but only fit_samples={int(fit_arr.shape[0])} "
            "domain samples were available; expect empty clusters.",
            flush=True,
        )

    # Build boundary point sets for guess (subsampled).
    a_pts: List[np.ndarray] = []
    b_pts: List[np.ndarray] = []
    for z, in_a, in_b in zip(z_trajs_use, in_a_trajs_use, in_b_trajs_use):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2:
            continue
        z_use = z[:, :dim_i]
        ia = np.nonzero(np.asarray(in_a, dtype=bool).reshape(-1))[0]
        ib = np.nonzero(np.asarray(in_b, dtype=bool).reshape(-1))[0]
        if ia.size:
            a_pts.append(np.asarray(z_use[ia], dtype=np.float64))
        if ib.size:
            b_pts.append(np.asarray(z_use[ib], dtype=np.float64))
    A = np.concatenate(a_pts, axis=0) if a_pts else np.zeros((0, dim_i), dtype=np.float64)
    B = np.concatenate(b_pts, axis=0) if b_pts else np.zeros((0, dim_i), dtype=np.float64)
    # Guardrail: committor requires both boundaries to be present.
    # If either is empty, the forward committor is ill-defined and solvers tend to fail.
    if int(A.shape[0]) == 0 or int(B.shape[0]) == 0:
        print(
            f"[Committor][DGA] Warning: empty boundary set during fit (nA={int(A.shape[0])}, nB={int(B.shape[0])}); "
            "skipping committor model fit.",
            flush=True,
        )
        return None
    if A.shape[0] > max_boundary_points:
        keep = rng.choice(int(A.shape[0]), size=int(max_boundary_points), replace=False)
        A = np.asarray(A[keep], dtype=np.float64)
    if B.shape[0] > max_boundary_points:
        keep = rng.choice(int(B.shape[0]), size=int(max_boundary_points), replace=False)
        B = np.asarray(B[keep], dtype=np.float64)

    # Apply the same normalization used for k-means fitting.
    if int(A.shape[0]) > 0:
        A = (np.asarray(A, dtype=np.float64) - mu[None, :]) / std[None, :]
    if int(B.shape[0]) > 0:
        B = (np.asarray(B, dtype=np.float64) - mu[None, :]) / std[None, :]

    nn_a = None
    nn_b = None
    if A.shape[0] >= 1 and B.shape[0] >= 1:
        nn_a = NearestNeighbors(n_neighbors=1)
        nn_a.fit(A)
        nn_b = NearestNeighbors(n_neighbors=1)
        nn_b.fit(B)

    # Backend selection:
    # - auto: try extq when importable; fall back to NumPy/SciPy on failure.
    # - extq: force an extq attempt (but still falls back on failure).
    # - numpy: never try extq (use the NumPy/SciPy port).
    backend_s = str(backend or "auto").strip().lower()
    if backend_s not in {"auto", "extq", "numpy"}:
        backend_s = "auto"
    force_extq = backend_s == "extq"
    use_extq = backend_s != "numpy"

    def _matrix_rank_warning_cls() -> type[Warning]:
        try:
            return scipy.sparse.linalg.MatrixRankWarning  # type: ignore[attr-defined]
        except Exception:
            return UserWarning

    def _attempt_fit(*, k: int) -> Tuple[Optional[DGAKMeansCommittorModel], Dict[str, Any]]:
        """Fit DGA committor for a particular k-means basis size `k`."""
        nonlocal use_extq
        diag: Dict[str, Any] = {"K": int(k)}
        diag["backend_requested"] = str(backend_s)

        k_i = int(k)
        if k_i <= 1:
            return None, diag
        if int(fit_arr.shape[0]) < 2:
            return None, diag
        if use_fixed_kmeans and fixed_kmeans is not None:
            try:
                k_i = int(getattr(fixed_kmeans, "n_clusters", k_i) or k_i)
            except Exception:
                pass
            diag["K_fixed"] = int(k_i)
        if (not use_fixed_kmeans) and k_i > int(fit_arr.shape[0]):
            k_i = int(max(2, min(k_i, int(fit_arr.shape[0]))))
            diag["K_adjusted"] = int(k_i)

        if use_fixed_kmeans and fixed_kmeans is not None:
            kmeans = fixed_kmeans
        else:
            kmeans = MiniBatchKMeans(n_clusters=int(k_i), random_state=int(seed))
            kmeans.fit(fit_arr)

        # Build basisL (unmasked) and basisF (masked to domain) per trajectory.
        basisL: List[scipy.sparse.spmatrix] = []
        basisF: List[scipy.sparse.spmatrix] = []
        labels_list: List[np.ndarray] = []
        in_domain_list: List[np.ndarray] = []
        guess_list: List[np.ndarray] = []
        nn_a_use = nn_a
        nn_b_use = nn_b

        for z, in_a, in_b in zip(z_trajs_use, in_a_trajs_use, in_b_trajs_use):
            z = np.asarray(z, dtype=np.float64)
            if z.ndim != 2:
                continue
            T = int(z.shape[0])
            if T <= lag_i:
                continue

            ia = np.asarray(in_a, dtype=bool).reshape(-1)
            ib = np.asarray(in_b, dtype=bool).reshape(-1)
            if ia.shape[0] != T or ib.shape[0] != T:
                continue
            dom = ~(ia | ib)

            z_use_raw = z[:, :dim_i]
            z_use = (np.asarray(z_use_raw, dtype=np.float64) - mu[None, :]) / std[None, :]
            labels = np.asarray(kmeans.predict(np.asarray(z_use, dtype=np.float64)), dtype=np.int64).reshape(-1)
            rows = np.arange(T, dtype=np.int64)
            cols = labels
            data = np.ones(T, dtype=np.float64)
            bL = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(T, int(k_i)))
            bF = scipy.sparse.diags(dom.astype(np.float64), format="csr") @ bL

            basisL.append(bL)
            basisF.append(bF)
            labels_list.append(labels.astype(np.int64, copy=False))
            in_domain_list.append(dom.astype(bool, copy=False))

            # Distance-based guess in VAC space when possible.
            if nn_a_use is None or nn_b_use is None:
                g = ib.astype(np.float64, copy=False)
                g[ia] = 0.0
                g[ib] = 1.0
            else:
                da = nn_a_use.kneighbors(z_use, n_neighbors=1, return_distance=True)[0].reshape(-1)
                db = nn_b_use.kneighbors(z_use, n_neighbors=1, return_distance=True)[0].reshape(-1)
                da2 = da * da
                db2 = db * db
                denom = da2 + db2 + float(eps)
                g = da2 / denom
                g = np.where(np.isfinite(g), g, 0.0)
                g = np.clip(g, 0.0, 1.0)
                g[ia] = 0.0
                g[ib] = 1.0
            guess_list.append(np.asarray(g, dtype=np.float64))

        if not basisL:
            return None, diag

        # Estimate invariant weights on the unmasked basis (prefer extq).
        weights: List[np.ndarray] = []
        backend = "numpy"
        extq = _get_extq(force=force_extq)
        if use_extq and extq is not None:
            extq_dga, _ = extq
            reweight_fn = getattr(extq_dga, "reweight", None)
            if callable(reweight_fn):
                try:
                    with warnings.catch_warnings():
                        with _silence_output_fds(1, 2):
                            w_out = reweight_fn(basisL, lag=lag_i, maxlag=lag_i)
                    weights = [np.asarray(w, dtype=np.float64).reshape(-1) for w in w_out]
                    backend = "extq"
                except Exception as exc:
                    print(
                        f"[Committor][DGA] Note: extq.reweight failed ({exc}); falling back to NumPy/SciPy weights.",
                        flush=True,
                    )
                    if not force_extq:
                        _disable_extq_runtime(f"runtime:reweight:{type(exc).__name__}")
                    use_extq = False
                    weights = []
                    backend = "numpy"
        if not weights:
            try:
                weights = dga_reweight(basisL, lag=lag_i, maxlag=lag_i)
            except Exception:
                weights = []
                for bL in basisL:
                    w = np.ones(int(bL.shape[0]), dtype=np.float64)
                    w[-lag_i:] = 0.0
                    weights.append(w)
        weights = _sanitize_weights(weights, basis=basisL, lag=lag_i)

        # Orthogonalize basisF w.r.t. weights. For indicator bases this reduces to per-cluster scaling.
        denom = float(sum(float(np.sum(w)) for w in weights)) if weights else 0.0
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        counts = np.zeros((int(k_i),), dtype=np.float64)
        for bF, w in zip(basisF, weights):
            try:
                counts += np.asarray(bF.T @ np.asarray(w, dtype=np.float64).reshape(-1), dtype=np.float64).reshape(-1)
            except Exception:
                continue
        p_raw = counts / denom
        p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)

        # Guardrail: cap the orthogonalization scaling for very rare clusters to reduce
        # ill-conditioning (rare/disconnected clusters are common early in training).
        max_basis_scale = 100.0
        p_floor = max(float(eps), 1.0 / float(max_basis_scale * max_basis_scale))
        n_floor = int(np.count_nonzero(p_raw <= float(p_floor)))
        diag["n_floor_clusters"] = int(n_floor)
        p = np.clip(p_raw, float(p_floor), None)
        basis_scale = (1.0 / np.sqrt(p)).astype(np.float64, copy=False)
        D = scipy.sparse.diags(basis_scale, format="csr")
        basis_d = [bF @ D for bF in basisF]

        # Solve for committor coefficients (prefer extq.forward_committor).
        coeffs = np.zeros((int(k_i),), dtype=np.float64)
        used_extq_committor = False
        solve_info: Optional[Any] = None
        solve_failed = False
        solve_exc_s: str = ""
        if use_extq and extq is not None:
            extq_dga, _ = extq
            fc_fn = getattr(extq_dga, "forward_committor", None)
            if callable(fc_fn):
                try:
                    with warnings.catch_warnings():
                        with _silence_output_fds(1, 2):
                            q_trajs = fc_fn(basis_d, weights, in_domain_list, guess_list, lag=lag_i)
                    q_trajs = [np.asarray(q, dtype=np.float64).reshape(-1) for q in q_trajs]
                    # Basic sanity: q must be finite; otherwise treat as failure.
                    ok_q = True
                    for q, g, dom in zip(q_trajs, guess_list, in_domain_list):
                        try:
                            qv = np.asarray(q, dtype=np.float64).reshape(-1)
                            gv = np.asarray(g, dtype=np.float64).reshape(-1)
                            dv = np.asarray(dom, dtype=bool).reshape(-1)
                            if qv.size == 0 or qv.size != gv.size or qv.size != dv.size:
                                ok_q = False
                                break
                            if not np.all(np.isfinite(qv)):
                                ok_q = False
                                break
                        except Exception:
                            ok_q = False
                            break
                    if not ok_q:
                        raise ValueError("extq.forward_committor returned invalid q trajectories.")
                    # Recover per-cluster coefficients from q = guess + basis_scale[label]*coeff[label].
                    sum_num = np.zeros((int(k_i),), dtype=np.float64)
                    sum_den = np.zeros((int(k_i),), dtype=np.float64)
                    for z, dom, g, q in zip(z_trajs_use, in_domain_list, guess_list, q_trajs):
                        z = np.asarray(z, dtype=np.float64)
                        if z.ndim != 2:
                            continue
                        T = int(z.shape[0])
                        if int(dom.shape[0]) != T or int(g.shape[0]) != T or int(q.shape[0]) != T:
                            continue
                        if T == 0:
                            continue
                        z_use_raw = np.asarray(z[:, :dim_i], dtype=np.float64)
                        z_use = (z_use_raw - mu[None, :]) / std[None, :]
                        labels = np.asarray(kmeans.predict(np.asarray(z_use, dtype=np.float64)), dtype=np.int64).reshape(-1)
                        idx = np.nonzero(dom)[0].astype(np.int64)
                        if idx.size == 0:
                            continue
                        lab = labels[idx]
                        diff = (q[idx] - np.asarray(g, dtype=np.float64).reshape(-1)[idx]) / basis_scale[lab]
                        sum_num += np.bincount(lab, weights=np.asarray(diff, dtype=np.float64), minlength=int(k_i))
                        sum_den += np.bincount(lab, minlength=int(k_i)).astype(np.float64)
                    coeffs = np.divide(sum_num, sum_den, out=np.zeros_like(sum_num), where=sum_den > 0.0)
                    used_extq_committor = True
                    backend = "extq"
                except Exception as exc:
                    print(
                        f"[Committor][DGA] Note: extq.forward_committor failed ({exc}); falling back to NumPy/SciPy committor.",
                        flush=True,
                    )
                    if not force_extq:
                        _disable_extq_runtime(f"runtime:committor:{type(exc).__name__}")
                    use_extq = False
                    used_extq_committor = False

        if not used_extq_committor:
            try:
                coeffs, solve_info = dga_forward_committor_coeffs(
                    basis_d,
                    weights,
                    in_domain_list,
                    guess_list,
                    lag=lag_i,
                    return_info=True,
                )
                coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
                if coeffs.shape[0] != int(k_i):
                    raise ValueError("Coefficient shape mismatch.")
                backend = "numpy"
            except Exception as exc:
                coeffs = np.zeros((int(k_i),), dtype=np.float64)
                backend = "numpy"
                solve_failed = True
                solve_exc_s = str(exc)

        diag["backend"] = str(backend)
        diag["solve_failed"] = bool(solve_failed)
        if solve_info is not None:
            try:
                diag["solve_method"] = str(getattr(solve_info, "method", "unknown"))
                diag["solve_ridge_eps"] = float(getattr(solve_info, "ridge_eps", 0.0) or 0.0)
            except Exception:
                pass
        elif solve_failed:
            diag["solve_method"] = "failed"
            diag["solve_note"] = solve_exc_s[:200]

        # Optional fallback: if the DGA linear solve is ill-conditioned or fails, try a
        # discrete MSM committor on the k-means labels (often more stable early on).
        use_discrete_proxy = False
        try:
            solve_method = str(diag.get("solve_method", "") or "")
            dga_unstable = bool(solve_failed) or (str(backend) != "extq" and solve_method not in ("spsolve", "dense_solve"))
            if dga_unstable and labels_list:
                disc = fit_vamp_discrete_committor(
                    labels_list,
                    in_a_trajs=in_a_trajs_use,
                    in_b_trajs=in_b_trajs_use,
                    n_states=int(k_i),
                    lag=int(lag_i),
                    prior_alpha=1e-3,
                )
                if not bool(getattr(disc, "degenerate", False)):
                    coeffs = np.asarray(getattr(disc, "q_state", np.zeros((int(k_i),), dtype=np.float64)), dtype=np.float64).reshape(-1)
                    if coeffs.shape[0] == int(k_i):
                        basis_scale = np.ones((int(k_i),), dtype=np.float64)
                        nn_a_use = None
                        nn_b_use = None
                        backend = "kmeans_discrete_proxy"
                        diag["backend"] = str(backend)
                        diag["solve_method"] = "discrete_msm"
                        diag["solve_failed"] = False
                        use_discrete_proxy = True
        except Exception:
            use_discrete_proxy = False

        model = DGAKMeansCommittorModel(
            kmeans=kmeans,
            nn_a=nn_a_use,
            nn_b=nn_b_use,
            coeffs=coeffs,
            basis_scale=basis_scale,
            dim=dim_i,
            lag=lag_i,
            delay_n_embed=int(delay_n),
            delay_delta=int(delay_delta_i),
            eps=float(eps),
            backend=str(backend),
            z_mu=np.asarray(mu, dtype=np.float64).reshape(-1),
            z_std=np.asarray(std, dtype=np.float64).reshape(-1),
        )

        # Diagnostics + degenerate-committor detection on the training set.
        solve_method = str(diag.get("solve_method", "") or "")
        solve_ridge_eps = float(diag.get("solve_ridge_eps", 0.0) or 0.0)
        model.solve_method = solve_method
        model.solve_ridge_eps = solve_ridge_eps

        q_dom_vals: List[np.ndarray] = []
        if use_discrete_proxy:
            try:
                q_state = np.asarray(coeffs, dtype=np.float64).reshape(-1)
                for lab, dom in zip(labels_list, in_domain_list):
                    lab_i = np.asarray(lab, dtype=np.int64).reshape(-1)
                    dom_b = np.asarray(dom, dtype=bool).reshape(-1)
                    n = int(min(int(lab_i.shape[0]), int(dom_b.shape[0])))
                    if n <= 0 or not bool(np.any(dom_b[:n])):
                        continue
                    lab_i = np.clip(lab_i[:n], 0, max(0, int(q_state.shape[0]) - 1))
                    qv = q_state[lab_i]
                    q_dom_vals.append(np.asarray(qv[dom_b[:n]], dtype=np.float64))
            except Exception:
                q_dom_vals = []
        else:
            try:
                for bF, g, dom in zip(basisF, guess_list, in_domain_list):
                    qv = np.asarray((bF @ D) @ coeffs, dtype=np.float64).reshape(-1) + np.asarray(g, dtype=np.float64).reshape(-1)
                    qv = np.where(np.isfinite(qv), qv, np.asarray(g, dtype=np.float64).reshape(-1))
                    qv = np.clip(qv, 0.0, 1.0)
                    dom_b = np.asarray(dom, dtype=bool).reshape(-1)
                    if qv.size and dom_b.size == qv.size and bool(np.any(dom_b)):
                        q_dom_vals.append(np.asarray(qv[dom_b], dtype=np.float64))
            except Exception:
                q_dom_vals = []

        if q_dom_vals:
            q_dom = np.concatenate(q_dom_vals, axis=0)
            try:
                model.q_domain_std = float(np.std(q_dom))
                diag["q_domain_std"] = float(model.q_domain_std)
                diag["q_domain_min"] = float(np.min(q_dom))
                diag["q_domain_max"] = float(np.max(q_dom))
                diag["q_domain_n"] = int(q_dom.shape[0])
            except Exception:
                pass
            try:
                q_range = float(diag.get("q_domain_max", 0.0)) - float(diag.get("q_domain_min", 0.0))
                q_std = float(diag.get("q_domain_std", float("nan")))
                n_dom = int(diag.get("q_domain_n", 0) or 0)
                # Mark as degenerate only when we have a reasonable number of domain frames.
                if n_dom >= 50 and (not np.isfinite(q_std) or q_std < 1e-3 or q_range < 5e-2):
                    model.degenerate = True
                    diag["degenerate"] = True
            except Exception:
                pass

        return model, diag

    if use_fixed_kmeans:
        model, _diag = _attempt_fit(k=int(K))
        return model

    # Stronger guardrail: if the committor solve becomes singular/regularized, retry with fewer clusters.
    max_attempts = 6
    last_model: Optional[DGAKMeansCommittorModel] = None
    last_diag: Dict[str, Any] = {}
    k_try = int(K)
    for attempt in range(int(max_attempts)):
        model, diag = _attempt_fit(k=k_try)
        if model is None:
            return None
        last_model = model
        last_diag = diag

        solve_method = str(diag.get("solve_method", "") or "")
        stable_solve = (diag.get("backend") in {"extq", "kmeans_discrete_proxy"}) or (solve_method in ("spsolve", "dense_solve", "discrete_msm"))
        stable = bool(stable_solve) and not bool(diag.get("degenerate", False))
        if stable:
            return model

        if attempt + 1 >= int(max_attempts):
            break

        k_next = int(max(2, int(math.floor(float(k_try) * 0.8))))
        if k_next >= int(k_try):
            k_next = int(k_try) - 1
        if k_next < 2:
            break

        print(
            f"[Committor][DGA] Warning: ill-conditioned committor solve (K={int(k_try)}, method={solve_method}); "
            f"retrying fit with K={int(k_next)}.",
            flush=True,
        )
        k_try = int(k_next)

    # Fall back to the last fit (possibly regularized) rather than failing outright.
    if last_model is not None:
        if str(last_diag.get("solve_method", "") or "") and str(last_diag.get("backend", "numpy")) != "extq":
            print(
                f"[Committor][DGA] Note: using last-fit committor coefficients despite instability "
                f"(K={int(last_diag.get('K', k_try))}, method={str(last_diag.get('solve_method', 'unknown'))}).",
                flush=True,
            )
        return last_model
    return None


@dataclass(frozen=True)
class VampDiscreteCommittorModel:
    """Discrete committor on VAMP argmax microstates ("softmax clusters")."""

    q_state: np.ndarray  # shape (K,)
    a_states: np.ndarray  # shape (K,) bool
    b_states: np.ndarray  # shape (K,) bool
    lag: int = 1
    backend: str = "vamp_discrete"
    degenerate: bool = False

    def predict_q(self, x: np.ndarray, *, in_a: Optional[np.ndarray] = None, in_b: Optional[np.ndarray] = None) -> np.ndarray:
        q, _ = self.predict_q_and_labels(x, in_a=in_a, in_b=in_b)
        return q

    def predict_q_and_labels(
        self,
        x: np.ndarray,
        *,
        in_a: Optional[np.ndarray] = None,
        in_b: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x)
        if x_arr.ndim == 2:
            labels = np.argmax(x_arr, axis=1).astype(np.int64, copy=False)
        else:
            labels = np.asarray(x_arr, dtype=np.int64).reshape(-1)
        K = int(self.q_state.shape[0])
        labels = np.clip(labels, 0, max(0, K - 1))
        q = np.asarray(self.q_state, dtype=np.float64).reshape(-1)[labels]
        q = np.where(np.isfinite(q), q, 0.0)
        q = np.clip(q, 0.0, 1.0)
        if in_a is not None:
            ia = np.asarray(in_a, dtype=bool).reshape(-1)
            n = int(min(int(q.shape[0]), int(ia.shape[0])))
            if n > 0:
                q[:n][ia[:n]] = 0.0
        if in_b is not None:
            ib = np.asarray(in_b, dtype=bool).reshape(-1)
            n = int(min(int(q.shape[0]), int(ib.shape[0])))
            if n > 0:
                q[:n][ib[:n]] = 1.0
        return q.astype(np.float64, copy=False), labels.astype(np.int64, copy=False)


def fit_vamp_discrete_committor(
    label_trajs: Sequence[np.ndarray],
    *,
    in_a_trajs: Sequence[np.ndarray],
    in_b_trajs: Sequence[np.ndarray],
    n_states: int,
    lag: int,
    prior_alpha: float = 1e-3,
) -> VampDiscreteCommittorModel:
    """
    Fit a discrete committor on VAMP argmax microstates.

    This estimates a state-space committor q(s) by:
      1) building a lagged transition matrix from labels,
      2) selecting absorbing state sets A/B from boundary masks,
      3) solving (I - P_OO) q_O = P_OB * 1.
    """

    K = int(n_states)
    lag_i = int(lag)
    if K <= 1:
        return VampDiscreteCommittorModel(
            q_state=np.zeros((max(0, K),), dtype=np.float64),
            a_states=np.zeros((max(0, K),), dtype=bool),
            b_states=np.zeros((max(0, K),), dtype=bool),
            lag=max(1, lag_i),
            degenerate=True,
        )
    lag_i = max(1, lag_i)

    counts = np.zeros((int(K), int(K)), dtype=np.float64)
    boundary_a_counts = np.zeros((int(K),), dtype=np.float64)
    boundary_b_counts = np.zeros((int(K),), dtype=np.float64)

    for labels_raw, ia_raw, ib_raw in zip(label_trajs, in_a_trajs, in_b_trajs):
        labels = np.asarray(labels_raw, dtype=np.int64).reshape(-1)
        ia = np.asarray(ia_raw, dtype=bool).reshape(-1)
        ib = np.asarray(ib_raw, dtype=bool).reshape(-1)
        n = int(min(int(labels.shape[0]), int(ia.shape[0]), int(ib.shape[0])))
        if n <= lag_i:
            continue
        labels = np.clip(labels[:n], 0, int(K) - 1)
        ia = ia[:n]
        ib = ib[:n]

        # Boundary hit counts per state (used to pick absorbing sets).
        try:
            if np.any(ia):
                np.add.at(boundary_a_counts, labels[ia], 1.0)
            if np.any(ib):
                np.add.at(boundary_b_counts, labels[ib], 1.0)
        except Exception:
            pass

        a = labels[: int(n - lag_i)]
        b = labels[int(lag_i) : int(n)]
        idx = a.astype(np.int64) * int(K) + b.astype(np.int64)
        flat = np.bincount(idx, minlength=int(K) * int(K)).astype(np.float64, copy=False)
        counts += flat.reshape((int(K), int(K)))

    # Assign absorbing state sets from boundary membership. If a state is observed in both
    # boundaries, assign it to the more frequently hit side (ties broken globally).
    a_states = boundary_a_counts > boundary_b_counts
    b_states = boundary_b_counts > boundary_a_counts
    tie = (boundary_a_counts > 0) & (boundary_b_counts > 0) & (boundary_a_counts == boundary_b_counts)
    if np.any(tie):
        if float(np.sum(boundary_a_counts)) >= float(np.sum(boundary_b_counts)):
            a_states = a_states | tie
        else:
            b_states = b_states | tie
    both = a_states & b_states
    if np.any(both):
        # Shouldn't happen, but be conservative.
        b_states = b_states & ~both

    # If A/B are empty, the committor is undefined. Return a degenerate model (all zeros).
    if (not np.any(a_states)) or (not np.any(b_states)):
        q_state = np.zeros((int(K),), dtype=np.float64)
        q_state[np.asarray(b_states, dtype=bool)] = 1.0
        return VampDiscreteCommittorModel(
            q_state=q_state,
            a_states=np.asarray(a_states, dtype=bool),
            b_states=np.asarray(b_states, dtype=bool),
            lag=int(lag_i),
            degenerate=True,
        )

    alpha = float(prior_alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        alpha = 1e-6

    # Build a row-stochastic transition matrix.
    c = np.asarray(counts, dtype=np.float64) + alpha
    row_sum = np.sum(c, axis=1)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    P = c / row_sum[:, None]

    # Enforce absorbing boundaries.
    for s in np.nonzero(a_states | b_states)[0].tolist():
        P[int(s), :] = 0.0
        P[int(s), int(s)] = 1.0

    idx_o = np.nonzero(~(a_states | b_states))[0]
    idx_b = np.nonzero(b_states)[0]
    q_state = np.zeros((int(K),), dtype=np.float64)
    q_state[np.asarray(b_states, dtype=bool)] = 1.0

    degenerate = False
    if idx_o.size:
        P_oo = P[np.ix_(idx_o, idx_o)]
        P_ob = P[np.ix_(idx_o, idx_b)]
        rhs = np.sum(P_ob, axis=1)
        A = np.eye(int(idx_o.size), dtype=np.float64) - np.asarray(P_oo, dtype=np.float64)
        try:
            q_o = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            try:
                ridge = 1e-6
                q_o = np.linalg.solve(A + ridge * np.eye(int(idx_o.size), dtype=np.float64), rhs)
            except Exception:
                q_o = np.linalg.lstsq(A, rhs, rcond=None)[0]
                degenerate = True
        q_o = np.where(np.isfinite(q_o), q_o, 0.0)
        q_o = np.clip(q_o, 0.0, 1.0)
        q_state[idx_o] = q_o

    # Flag near-constant solutions as degenerate (common with little boundary data).
    try:
        if float(np.std(q_state)) < 1e-3:
            degenerate = True
    except Exception:
        pass

    return VampDiscreteCommittorModel(
        q_state=q_state.astype(np.float64, copy=False),
        a_states=np.asarray(a_states, dtype=bool),
        b_states=np.asarray(b_states, dtype=bool),
        lag=int(lag_i),
        degenerate=bool(degenerate),
    )
