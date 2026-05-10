"""Components subpackage — lazy imports to avoid pulling in openmm at import time."""


def __getattr__(name: str):
    _lazy = {
        "GeomVecManager": ".geom2vec_manager",
        "LOOVAMPRewards": "._reexports",
        "RetrievalManager": ".retrieval_manager",
        "EpochCoordinator": ".epoch_coordinator",
        "SamplingOrchestrator": ".sampling_orchestrator",
        "save_torsion_ar_policy": ".policy_io",
        "graph_algorithms": "._reexports",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        return getattr(mod, name, mod)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
