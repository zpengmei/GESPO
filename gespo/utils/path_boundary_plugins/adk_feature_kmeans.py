from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from gespo.utils.geometry import calculate_rmsd

try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
except Exception:  # pragma: no cover
    MiniBatchKMeans = None


@dataclass(frozen=True)
class _AdkGroups:
    core_lid: np.ndarray
    core1: np.ndarray
    nmp: np.ndarray
    core2: np.ndarray
    lid: np.ndarray
    res145: int
    res52: int
    res169: int
    res55: int


@dataclass
class _FittedBoundaryModel:
    kmeans: Any
    mu: np.ndarray
    std: np.ndarray
    a_clusters: np.ndarray
    b_clusters: np.ndarray
    groups: _AdkGroups


_MODEL_CACHE: Dict[str, _FittedBoundaryModel] = {}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(c, dtype=np.float64).reshape(-1, 3)
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    denom_safe = np.where(denom > 1e-12, denom, 1.0)
    cosang = np.sum(v1 * v2, axis=1) / denom_safe
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    ang = np.where(denom > 1e-12, ang, 0.0)
    return ang.astype(np.float32, copy=False)


def _centroid(ca_nm: np.ndarray, token_idx: np.ndarray) -> np.ndarray:
    ca_nm = np.asarray(ca_nm, dtype=np.float64)
    token_idx = np.asarray(token_idx, dtype=np.int64).reshape(-1)
    if ca_nm.ndim != 3 or ca_nm.shape[-1] != 3:
        raise ValueError(f"ca_nm must be (T, tokens, 3), got {ca_nm.shape}")
    if token_idx.size == 0:
        return np.zeros((int(ca_nm.shape[0]), 3), dtype=np.float32)
    token_idx = token_idx[(token_idx >= 0) & (token_idx < int(ca_nm.shape[1]))]
    if token_idx.size == 0:
        return np.zeros((int(ca_nm.shape[0]), 3), dtype=np.float32)
    return ca_nm[:, token_idx, :].mean(axis=1).astype(np.float32, copy=False)


def _distance(ca_nm: np.ndarray, a: int, b: int) -> np.ndarray:
    ca_nm = np.asarray(ca_nm, dtype=np.float64)
    if ca_nm.ndim != 3 or ca_nm.shape[-1] != 3:
        raise ValueError(f"ca_nm must be (T, tokens, 3), got {ca_nm.shape}")
    T, n_tok, _ = ca_nm.shape
    a_i = int(a)
    b_i = int(b)
    if a_i < 0 or b_i < 0 or a_i >= int(n_tok) or b_i >= int(n_tok) or a_i == b_i:
        return np.zeros((int(T),), dtype=np.float32)
    return np.linalg.norm(ca_nm[:, a_i, :] - ca_nm[:, b_i, :], axis=1).astype(np.float32, copy=False)


def _get_cfg(config: Any, name: str, default: Any) -> Any:
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def _build_adk_groups_from_endpoint_pdb(
    pdb_path: str,
    *,
    n_tokens: int,
    token_mask: Optional[np.ndarray],
) -> _AdkGroups:
    try:
        import mdtraj as md  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mdtraj is required for ADK boundary plugin.") from exc

    tr = md.load(str(pdb_path))
    residues = list(tr.topology.residues)
    if not residues:
        raise ValueError("No residues in endpoint PDB.")

    # Token order in the geom2vec cache is residue order; map residue numbers via resSeq when present.
    resseq: list[int] = []
    for r in residues:
        try:
            rs = getattr(r, "resSeq", None)
            rs_i = int(rs) if rs is not None else int(r.index)
        except Exception:
            rs_i = int(r.index)
        resseq.append(int(rs_i))

    # Guard against endpoint residue list mismatching token count. We only define
    # ADK groups over the overlapping prefix.
    n_use = int(min(int(n_tokens), len(resseq)))
    resseq = resseq[:n_use]
    mask = None
    if token_mask is not None:
        try:
            m = np.asarray(token_mask, dtype=bool).reshape(-1)
            if int(m.size) == int(n_tokens):
                mask = m[:n_use]
        except Exception:
            mask = None

    def _range(start: int, end: int) -> np.ndarray:
        start_i = int(min(start, end))
        end_i = int(max(start, end))
        idx = np.asarray([i for i, rs in enumerate(resseq) if start_i <= int(rs) <= end_i], dtype=np.int64)
        if mask is not None and idx.size:
            idx = idx[mask[idx]]
        return idx

    resseq_to_token: Dict[int, int] = {}
    for i, rs in enumerate(resseq):
        if int(rs) not in resseq_to_token:
            resseq_to_token[int(rs)] = int(i)

    def _res(resnum: int) -> int:
        target = int(resnum)
        if target in resseq_to_token:
            i = int(resseq_to_token[target])
            if mask is None or (0 <= i < int(mask.size) and bool(mask[i])):
                return i
        # Fallback: interpret as 0-based residue index.
        i = int(target)
        if 0 <= i < n_use and (mask is None or bool(mask[i])):
            return i
        i = int(target - 1)
        if 0 <= i < n_use and (mask is None or bool(mask[i])):
            return i
        return -1

    return _AdkGroups(
        core_lid=_range(115, 125),
        core1=_range(90, 100),
        nmp=_range(35, 55),
        core2=_range(179, 185),
        lid=_range(125, 153),
        res145=_res(145),
        res52=_res(52),
        res169=_res(169),
        res55=_res(55),
    )


def _rmsd_to_endpoint(
    ca_nm: np.ndarray, ref_nm: np.ndarray, *, token_mask: Optional[np.ndarray]
) -> np.ndarray:
    ca_nm = np.asarray(ca_nm, dtype=np.float64)
    ref_nm = np.asarray(ref_nm, dtype=np.float64)
    if ca_nm.ndim != 3 or ca_nm.shape[-1] != 3:
        raise ValueError(f"ca_nm must be (T, tokens, 3), got {ca_nm.shape}")
    if ref_nm.ndim != 2 or ref_nm.shape[-1] != 3:
        raise ValueError(f"ref_nm must be (tokens, 3), got {ref_nm.shape}")
    if int(ca_nm.shape[1]) != int(ref_nm.shape[0]):
        raise ValueError("RMSD-to-endpoint token mismatch.")
    mask = None
    if token_mask is not None:
        try:
            m = np.asarray(token_mask, dtype=bool).reshape(-1)
            if int(m.size) == int(ca_nm.shape[1]):
                mask = m
        except Exception:
            mask = None
    if mask is None:
        mask = np.ones((int(ca_nm.shape[1]),), dtype=bool)
    ca_use = ca_nm[:, mask, :]
    ref_use = ref_nm[mask, :]
    if int(ref_use.shape[0]) < 3:
        return np.full((int(ca_nm.shape[0]),), np.inf, dtype=np.float64)
    return calculate_rmsd(ca_use, ref_use, atom_indices=None).astype(np.float64, copy=False)


def _feature_matrix(
    *,
    ca_nm: np.ndarray,
    z_lin: Optional[np.ndarray],
    rmsd_a: np.ndarray,
    rmsd_b: np.ndarray,
    groups: _AdkGroups,
) -> np.ndarray:
    ca_nm = np.asarray(ca_nm, dtype=np.float64)
    T = int(ca_nm.shape[0])

    z = np.zeros((T, 2), dtype=np.float32)
    if z_lin is not None:
        try:
            z_arr = np.asarray(z_lin, dtype=np.float32)
            if z_arr.ndim == 2 and int(z_arr.shape[0]) == T and int(z_arr.shape[1]) >= 2:
                z[:, 0] = z_arr[:, 0]
                z[:, 1] = z_arr[:, 1]
        except Exception:
            pass

    core_lid = _centroid(ca_nm, groups.core_lid)
    core1 = _centroid(ca_nm, groups.core1)
    nmp = _centroid(ca_nm, groups.nmp)
    core2 = _centroid(ca_nm, groups.core2)
    lid = _centroid(ca_nm, groups.lid)

    theta_nmp = _angle(core_lid, core1, nmp)
    theta_lid = _angle(core2, core_lid, lid)
    dist_145_52 = _distance(ca_nm, groups.res145, groups.res52)
    dist_169_55 = _distance(ca_nm, groups.res169, groups.res55)

    ra = np.asarray(rmsd_a, dtype=np.float32).reshape(T)
    rb = np.asarray(rmsd_b, dtype=np.float32).reshape(T)
    feats = np.column_stack([z[:, 0], z[:, 1], theta_nmp, theta_lid, dist_145_52, dist_169_55, ra, rb]).astype(
        np.float32, copy=False
    )
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32, copy=False)
    return feats


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    Z = (X - mu) / std
    return Z.astype(np.float32, copy=False), mu.reshape(-1).astype(np.float32, copy=False), std.reshape(-1).astype(np.float32, copy=False)


def adk_feature_kmeans_boundaries(
    *,
    ca_coords: np.ndarray,
    endpoint_ca_ref_a: Optional[np.ndarray],
    endpoint_ca_ref_b: Optional[np.ndarray],
    cutoff_a_nm: float,
    cutoff_b_nm: float,
    token_mask: Optional[np.ndarray] = None,
    z_lin: Optional[np.ndarray] = None,
    fit_keys: Optional[Sequence[str]] = None,
    fit_nonce: int = 0,
    embedding_cache: Any = None,
    projector_linear: Any = None,
    config: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boundary plugin: k-means over [slow coords + ADK geometry + RMSDs], then take nearest ~10% clusters.

    Intended usage:
    - Set `--path-boundary-plugin /abs/.../adk_feature_kmeans.py:adk_feature_kmeans_boundaries`
    - Provide endpoint A/B via `--pdb`/`--pdb-b` (path sampling).
    - The training loop passes `fit_keys`, `embedding_cache`, and `projector_linear` so the plugin can
      fit (once per refresh) on cached history embeddings.
    """

    ca_nm = np.asarray(ca_coords, dtype=np.float64)
    if ca_nm.ndim != 3 or ca_nm.shape[-1] != 3:
        raise ValueError(f"ca_coords must be (T, tokens, 3) in nm, got {ca_nm.shape}")
    T, n_tok, _ = ca_nm.shape

    if endpoint_ca_ref_a is None or endpoint_ca_ref_b is None:
        raise ValueError("endpoint_ca_ref_a and endpoint_ca_ref_b are required for ADK boundary plugin.")
    ref_a = np.asarray(endpoint_ca_ref_a, dtype=np.float64)
    ref_b = np.asarray(endpoint_ca_ref_b, dtype=np.float64)
    if ref_a.shape != (int(n_tok), 3) or ref_b.shape != (int(n_tok), 3):
        raise ValueError("Endpoint CA references must have shape (tokens, 3) matching ca_coords.")

    # Fit config (reuse the existing VAC-kmeans knobs to avoid growing the CLI surface).
    Kb = int(_get_cfg(config, "path_boundary_vac_kmeans_n_clusters", 50) or 50)
    max_samples = int(_get_cfg(config, "path_boundary_vac_kmeans_max_fit_samples", 50000) or 50000)
    seed_b = int(_get_cfg(config, "path_boundary_vac_kmeans_seed", 42) or 42)
    pct_a = float(_get_cfg(config, "path_boundary_vac_kmeans_percent_a", 10.0) or 10.0)
    pct_b = float(_get_cfg(config, "path_boundary_vac_kmeans_percent_b", pct_a) or pct_a)
    pct_a = max(0.0, min(100.0, pct_a))
    pct_b = max(0.0, min(100.0, pct_b))

    # Interpret percent as "percent of clusters" (not frames), per ADK usage.
    n_pick_a = max(1, int(math.ceil((pct_a / 100.0) * float(Kb)))) if Kb > 0 else 1
    n_pick_b = max(1, int(math.ceil((pct_b / 100.0) * float(Kb)))) if Kb > 0 else 1

    # Model key: endpoints + nonce + fit params
    pdb_a = str(_get_cfg(config, "pdb_a", "") or _get_cfg(config, "pdb", "") or "")
    pdb_b = str(_get_cfg(config, "pdb_b", "") or "")
    sig = f"adk_feature_kmeans|{pdb_a}|{pdb_b}|nonce={int(fit_nonce)}|K={int(Kb)}|seed={int(seed_b)}|max={int(max_samples)}|pa={pct_a:.3f}|pb={pct_b:.3f}"

    model = _MODEL_CACHE.get(sig)
    if model is None:
        if MiniBatchKMeans is None:
            raise RuntimeError("scikit-learn is required for k-means boundaries (MiniBatchKMeans unavailable).")
        if embedding_cache is None or projector_linear is None or not fit_keys:
            # No fit context yet; fall back to RMSD cutoffs.
            ra = _rmsd_to_endpoint(ca_nm, ref_a, token_mask=token_mask)
            rb = _rmsd_to_endpoint(ca_nm, ref_b, token_mask=token_mask)
            in_a = ra <= float(cutoff_a_nm)
            in_b = rb <= float(cutoff_b_nm)
            both = in_a & in_b
            if np.any(both):
                choose_a = ra < rb
                in_a = in_a & ~both
                in_b = in_b & ~both
                in_a[both] = choose_a[both]
                in_b[both] = ~choose_a[both]
            return in_a.astype(bool, copy=False), in_b.astype(bool, copy=False), ra, rb

        # Fit on cached history.
        groups = _build_adk_groups_from_endpoint_pdb(pdb_a or pdb_b, n_tokens=int(n_tok), token_mask=token_mask)

        feat_blocks: list[np.ndarray] = []
        ra_blocks: list[np.ndarray] = []
        rb_blocks: list[np.ndarray] = []
        collected = 0

        for key in list(fit_keys):
            if max_samples > 0 and collected >= int(max_samples):
                break
            try:
                graph, ca_aa_t = embedding_cache.load(str(key))
            except Exception:
                continue
            try:
                # Cached CA coords are in Å in the GESPO geom2vec cache; convert to nm.
                ca_aa = ca_aa_t.detach().cpu().numpy() if hasattr(ca_aa_t, "detach") else np.asarray(ca_aa_t)
                ca_nm_key = np.asarray(ca_aa, dtype=np.float64) * 0.1
                if ca_nm_key.ndim != 3 or ca_nm_key.shape[-1] != 3 or int(ca_nm_key.shape[1]) != int(n_tok):
                    continue
            except Exception:
                continue
            try:
                z_key = projector_linear.state_output_from_embeddings(graph, ca_aa_t)
            except Exception:
                z_key = None

            try:
                ra_key = _rmsd_to_endpoint(ca_nm_key, ref_a, token_mask=token_mask)
                rb_key = _rmsd_to_endpoint(ca_nm_key, ref_b, token_mask=token_mask)
            except Exception:
                continue

            feats_key = _feature_matrix(ca_nm=ca_nm_key, z_lin=z_key, rmsd_a=ra_key, rmsd_b=rb_key, groups=groups)
            n_here = int(feats_key.shape[0])
            if n_here <= 0:
                continue

            if max_samples > 0:
                remaining = int(max_samples) - collected
                if remaining <= 0:
                    break
                if n_here > remaining:
                    idx = np.linspace(0, n_here - 1, num=int(remaining), dtype=np.int64)
                    feats_key = feats_key[idx]
                    ra_key = ra_key[idx]
                    rb_key = rb_key[idx]
                    n_here = int(feats_key.shape[0])

            feat_blocks.append(feats_key)
            ra_blocks.append(np.asarray(ra_key, dtype=np.float64).reshape(-1))
            rb_blocks.append(np.asarray(rb_key, dtype=np.float64).reshape(-1))
            collected += int(n_here)

        if not feat_blocks:
            raise RuntimeError("No cached frames available to fit ADK k-means boundary model.")

        X = np.concatenate(feat_blocks, axis=0).astype(np.float32, copy=False)
        RA = np.concatenate(ra_blocks, axis=0).astype(np.float64, copy=False)
        RB = np.concatenate(rb_blocks, axis=0).astype(np.float64, copy=False)

        Z, mu, std = _standardize(X)
        kmeans = MiniBatchKMeans(n_clusters=int(Kb), random_state=int(seed_b), batch_size=2048, n_init="auto")
        labels = kmeans.fit_predict(Z)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        counts = np.bincount(labels, minlength=int(Kb)).astype(np.int64, copy=False)
        denom = np.where(counts > 0, counts.astype(np.float64), 1.0)
        mean_ra = np.bincount(labels, weights=np.asarray(RA, dtype=np.float64), minlength=int(Kb)) / denom
        mean_rb = np.bincount(labels, weights=np.asarray(RB, dtype=np.float64), minlength=int(Kb)) / denom
        frac_closer_a = (
            np.bincount(labels, weights=(np.asarray(RA) <= np.asarray(RB)).astype(np.float64), minlength=int(Kb)) / denom
        )
        side_a = (frac_closer_a >= 0.5) & (counts > 0)

        a_clusters = np.zeros((int(Kb),), dtype=bool)
        b_clusters = np.zeros((int(Kb),), dtype=bool)

        a_cand = np.nonzero(side_a)[0]
        if a_cand.size == 0:
            a_cand = np.nonzero(counts > 0)[0]
        a_order = a_cand[np.argsort(mean_ra[a_cand])]
        for k in a_order[: int(n_pick_a)].tolist():
            a_clusters[int(k)] = True

        b_cand = np.nonzero((~side_a) & (counts > 0))[0]
        if b_cand.size == 0:
            b_cand = np.nonzero(counts > 0)[0]
        b_order = b_cand[np.argsort(mean_rb[b_cand])]
        picked_b = 0
        for k in b_order.tolist():
            if picked_b >= int(n_pick_b):
                break
            if a_clusters[int(k)]:
                continue
            b_clusters[int(k)] = True
            picked_b += 1

        model = _FittedBoundaryModel(
            kmeans=kmeans,
            mu=np.asarray(mu, dtype=np.float32).reshape(-1),
            std=np.asarray(std, dtype=np.float32).reshape(-1),
            a_clusters=a_clusters,
            b_clusters=b_clusters,
            groups=groups,
        )
        _MODEL_CACHE[sig] = model

    # Inference for the current trajectory.
    ra = _rmsd_to_endpoint(ca_nm, ref_a, token_mask=token_mask)
    rb = _rmsd_to_endpoint(ca_nm, ref_b, token_mask=token_mask)
    X_now = _feature_matrix(ca_nm=ca_nm, z_lin=z_lin, rmsd_a=ra, rmsd_b=rb, groups=model.groups).astype(np.float32, copy=False)
    Z_now = (X_now - model.mu.reshape(1, -1)) / np.where(model.std.reshape(1, -1) > 1e-8, model.std.reshape(1, -1), 1.0)
    labels_now = np.asarray(model.kmeans.predict(Z_now), dtype=np.int64).reshape(int(T))
    in_a = np.asarray(model.a_clusters, dtype=bool)[labels_now]
    in_b = np.asarray(model.b_clusters, dtype=bool)[labels_now]

    both = in_a & in_b
    if np.any(both):
        choose_a = ra <= rb
        in_a = in_a & ~both
        in_b = in_b & ~both
        in_a[both] = choose_a[both]
        in_b[both] = ~choose_a[both]

    return in_a.astype(bool, copy=False), in_b.astype(bool, copy=False), ra, rb

