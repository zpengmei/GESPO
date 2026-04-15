"""Shim — geom2vec backbone moved to gespo.projections.geom2vec."""
from gespo.projections.geom2vec import *  # noqa: F401,F403
from gespo.projections.geom2vec import (
    EmbeddingCache, EmbeddingRecord,
    TimeLaggedIndexDataset,
    bootstrap_cache_from_dcd_dir, scan_dcd_files,
    infer_geom2vec_embeddings, infer_and_cache,
    Geom2VecVAMPProjector,
    Geom2VecTopologyCache, TorchGeom2VecTopologyCache,
    TrainedVAMPLobe, train_vamp_lobe_from_cache,
)
