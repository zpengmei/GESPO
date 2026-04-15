"""Pure graph algorithms for tube-cell connectivity analysis (path sampling)."""
from __future__ import annotations

import collections
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def shortest_path_len(
    *,
    adj: Sequence[Set[int]],
    sources: Sequence[int],
    targets: Set[int],
    overlay: Optional[Dict[int, List[int]]] = None,
) -> float:
    """Multi-source BFS shortest path length (in edges) on an undirected graph.

    Returns +inf if no path exists.
    """
    n = int(len(adj))
    if n <= 0 or not targets or not sources:
        return float("inf")
    src = [int(s) for s in sources if 0 <= int(s) < n]
    if not src:
        return float("inf")

    dist = -np.ones((n,), dtype=np.int64)
    dq = collections.deque()
    for s in src:
        dist[s] = 0
        dq.append(s)
    while dq:
        u = int(dq.popleft())
        du = int(dist[u])
        if u in targets:
            return float(du)
        neigh = adj[u]
        if neigh:
            for v in neigh:
                vi = int(v)
                if 0 <= vi < n and dist[vi] < 0:
                    dist[vi] = du + 1
                    dq.append(vi)
        if overlay is not None:
            extra = overlay.get(u)
            if extra:
                for v in extra:
                    vi = int(v)
                    if 0 <= vi < n and dist[vi] < 0:
                        dist[vi] = du + 1
                        dq.append(vi)
    return float("inf")


def component_ids(adj: Sequence[Set[int]]) -> np.ndarray:
    """Return component id per node for an undirected graph."""
    n = int(len(adj))
    comp = -np.ones((n,), dtype=np.int64)
    cid = 0

    for i in range(n):
        if comp[i] >= 0:
            continue
        comp[i] = cid
        dq = collections.deque([i])
        while dq:
            u = int(dq.popleft())
            for v in adj[u]:
                vi = int(v)
                if 0 <= vi < n and comp[vi] < 0:
                    comp[vi] = cid
                    dq.append(vi)
        cid += 1
    return comp


def reachable_nodes(
    *,
    adj: Sequence[Set[int]],
    sources: Sequence[int],
    overlay: Optional[Dict[int, List[int]]] = None,
) -> np.ndarray:
    """Return a boolean mask of nodes reachable from sources in an undirected graph."""
    n = int(len(adj))
    if n <= 0 or not sources:
        return np.zeros((max(0, n),), dtype=bool)
    src = [int(s) for s in sources if 0 <= int(s) < n]
    if not src:
        return np.zeros((n,), dtype=bool)

    seen = np.zeros((n,), dtype=bool)
    dq = collections.deque()
    for s in src:
        if not seen[s]:
            seen[s] = True
            dq.append(s)
    while dq:
        u = int(dq.popleft())
        neigh = adj[u]
        if neigh:
            for v in neigh:
                vi = int(v)
                if 0 <= vi < n and (not seen[vi]):
                    seen[vi] = True
                    dq.append(vi)
        if overlay is not None:
            extra = overlay.get(u)
            if extra:
                for v in extra:
                    vi = int(v)
                    if 0 <= vi < n and (not seen[vi]):
                        seen[vi] = True
                        dq.append(vi)
    return seen


def bin_gap(
    *,
    reachable_a: np.ndarray,
    reachable_b: np.ndarray,
    K: int,
    M: int,
) -> Tuple[int, int, int]:
    """Return (gap, a_max_bin, b_min_bin) from reachable masks.

    gap=0 means the reachable components overlap in committor-bin index.
    """
    K_i = int(max(1, int(K)))
    M_i = int(max(1, int(M)))
    n = int(min(int(reachable_a.shape[0]), int(reachable_b.shape[0])))
    if n <= 0:
        return int(M_i - 1), 0, int(M_i - 1)
    a_idx = np.nonzero(np.asarray(reachable_a[:n], dtype=bool))[0]
    b_idx = np.nonzero(np.asarray(reachable_b[:n], dtype=bool))[0]
    if a_idx.size == 0 or b_idx.size == 0:
        return int(M_i - 1), 0, int(M_i - 1)
    a_bins = (a_idx // K_i).astype(np.int64, copy=False)
    b_bins = (b_idx // K_i).astype(np.int64, copy=False)
    a_max = int(np.max(a_bins))
    b_min = int(np.min(b_bins))
    gap = int(max(0, b_min - a_max - 1))
    return gap, a_max, b_min


def committor_progress(q: np.ndarray) -> float:
    """Symmetric committor progress: max displacement from starting committor value."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size == 0:
        return 0.0
    q0 = float(q[0])
    qmax = float(np.max(q))
    qmin = float(np.min(q))
    return float(max(0.0, qmax - q0, q0 - qmin))


def max_flow_unit_cap(
    *,
    adj: Dict[int, set[int]],
    sources: List[int],
    targets: set[int],
    overlay: Optional[Dict[int, List[int]]],
    cap_max: int,
    ) -> int:
    """Approximate A↔B path diversity via unit-capacity max flow (edge-disjoint paths).

    Uses Dinic on an undirected unit-capacity graph (implemented as two directed unit edges).
    """
    if not sources or not targets:
        return 0
    cap_lim = int(max(1, int(cap_max))) if int(cap_max) > 0 else 1_000_000_000
    # Trivial overlap => already "inf" connected; cap at cap_lim.
    if any(int(s) in targets for s in sources):
        return int(cap_lim)

    # Build a compact node set to avoid allocating on large sparse IDs.
    nodes: set[int] = set(int(s) for s in sources)
    nodes.update(int(t) for t in targets)
    for u, nbrs in adj.items():
        if not nbrs:
            continue
        iu = int(u)
        if iu in nodes:
            nodes.update(int(v) for v in nbrs)
    if overlay:
        for u, nbrs in overlay.items():
            iu = int(u)
            nodes.add(iu)
            nodes.update(int(v) for v in nbrs)

    # Map node IDs to 0..N-1
    node_list = sorted(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    n_base = int(len(node_list))
    s_idx = n_base
    t_idx = n_base + 1
    N = n_base + 2

    class _E:
        __slots__ = ("to", "rev", "cap")

        def __init__(self, to: int, rev: int, cap: int):
            self.to = int(to)
            self.rev = int(rev)
            self.cap = int(cap)

    g: List[List[_E]] = [[] for _ in range(N)]

    def _add(u: int, v: int, cap: int) -> None:
        a = _E(v, len(g[v]), cap)
        b = _E(u, len(g[u]), 0)
        g[u].append(a)
        g[v].append(b)

    def _add_undirected(u: int, v: int) -> None:
        # unit capacity in each direction
        _add(u, v, 1)
        _add(v, u, 1)

    # Add undirected edges from base adj
    for u in nodes:
        iu = int(u)
        if iu not in adj:
            continue
        for v in adj[iu]:
            iv = int(v)
            if iv not in nodes:
                continue
            # Add each undirected edge once (u<v) to avoid doubling.
            if iu < iv:
                _add_undirected(idx[iu], idx[iv])
    # Add overlay edges
    if overlay:
        for u, nbrs in overlay.items():
            iu = int(u)
            if iu not in idx:
                continue
            for v in nbrs:
                iv = int(v)
                if iv not in idx:
                    continue
                if iu < iv:
                    _add_undirected(idx[iu], idx[iv])
                elif iv < iu:
                    _add_undirected(idx[iv], idx[iu])

    # Connect super source/sink
    for s in sources:
        if int(s) in idx:
            _add(s_idx, idx[int(s)], cap_lim)
    for t in targets:
        if int(t) in idx:
            _add(idx[int(t)], t_idx, cap_lim)

    level = [-1] * N
    it = [0] * N

    from collections import deque

    def bfs() -> bool:
        for i in range(N):
            level[i] = -1
        q = deque([s_idx])
        level[s_idx] = 0
        while q:
            u = q.popleft()
            for e in g[u]:
                if e.cap > 0 and level[e.to] < 0:
                    level[e.to] = level[u] + 1
                    q.append(e.to)
        return level[t_idx] >= 0

    def dfs(u: int, f: int) -> int:
        if u == t_idx:
            return f
        for i in range(it[u], len(g[u])):
            it[u] = i
            e = g[u][i]
            if e.cap <= 0 or level[e.to] != level[u] + 1:
                continue
            ret = dfs(e.to, min(f, e.cap))
            if ret > 0:
                e.cap -= ret
                g[e.to][e.rev].cap += ret
                return ret
        return 0

    flow = 0
    while flow < cap_lim and bfs():
        for i in range(N):
            it[i] = 0
        while flow < cap_lim:
            pushed = dfs(s_idx, cap_lim - flow)
            if pushed <= 0:
                break
            flow += pushed
    return int(flow)


