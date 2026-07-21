"""Multi-walker fan-out for the walker-based seed methods (metad / high-temp).

Each walker runs in its own ``spawn`` subprocess (independent CUDA context, one
GPU per walker via the ``device`` kwarg) and writes into ``<out_dir>/walker_i/``.
Metadynamics walkers share a single ``bias_dir`` so their Gaussians accumulate
collectively — exactly the parallel-metaD setup of the source launcher.

OpenMM is imported lazily inside the worker, so this module stays importable
without the scientific stack.
"""
from __future__ import annotations

import multiprocessing as _mp
from typing import Dict, List

from ._common import log, resolve_devices, walker_dirs


def _walker_worker(method: str, kwargs: Dict) -> None:
    """Entry point run in each spawned process: dispatch to the seed method."""
    if method == "metad":
        from .methods.metad import run_metad
        run_metad(**kwargs)
    elif method == "high-temp":
        from .methods.high_temp import run_high_temp
        run_high_temp(**kwargs)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"Unknown walker method '{method}'")


def _fan_out(method: str, kwargs_list: List[Dict]) -> None:
    """Run each kwargs set in its own spawned process; raise if any fail."""
    ctx = _mp.get_context("spawn")
    procs = []
    for kw in kwargs_list:
        p = ctx.Process(target=_walker_worker, args=(method, kw))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    failed = [i + 1 for i, p in enumerate(procs) if p.exitcode not in (0, None)]
    if failed:
        raise RuntimeError(f"Seed walker(s) {failed} exited with an error.")


def run_method(
    method: str,
    *,
    out_dir: str,
    n_walkers: int = 1,
    device: str = "cuda:0",
    bias_dir: str = "",
    method_kwargs: Dict,
) -> List[str]:
    """Run ``n_walkers`` of ``method`` into ``<out_dir>/walker_i/production.dcd``.

    Returns the list of expected production-DCD paths. A single walker runs
    in-process; multiple walkers fan out across ``device`` (a single device is
    reused; a comma-list is cycled). Metadynamics walkers share ``bias_dir``.
    """
    n = max(1, int(n_walkers))
    devices = resolve_devices(device, n)
    wdirs = walker_dirs(out_dir, n)
    shared_bias = str(bias_dir) if bias_dir else None

    per_walker = []
    for wd, dev in zip(wdirs, devices):
        kw = dict(method_kwargs)
        kw["out_dir"] = str(wd)
        kw["device"] = dev
        if method == "metad":
            kw["bias_dir"] = shared_bias  # None -> each walker gets its own bias/
        per_walker.append(kw)

    if n == 1:
        _walker_worker(method, per_walker[0])
    else:
        log(f"[seed] fanning out {n} {method} walkers across devices {devices}")
        _fan_out(method, per_walker)

    return [str(wd / "production.dcd") for wd in wdirs]
