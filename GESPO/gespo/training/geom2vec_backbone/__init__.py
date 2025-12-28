"""Geom2vec backbone integration utilities for GESPO.

This subpackage holds the planned building blocks for using geom2vec embeddings
as the default DeepTICA/VAMP backbone inside GESPO.
"""

from .cache import EmbeddingCache, EmbeddingRecord
from .dataset import TimeLaggedIndexDataset
from .bootstrap import bootstrap_cache_from_dcd_dir, scan_dcd_files
from .inference import infer_geom2vec_embeddings
from .pipeline import infer_and_cache
from .projector import Geom2VecVAMPProjector
from .topology import Geom2VecTopologyCache, TorchGeom2VecTopologyCache
from .vamp_trainer import TrainedVAMPLobe, train_vamp_lobe_from_cache

__all__ = [
    "EmbeddingCache",
    "EmbeddingRecord",
    "TimeLaggedIndexDataset",
    "bootstrap_cache_from_dcd_dir",
    "scan_dcd_files",
    "infer_geom2vec_embeddings",
    "infer_and_cache",
    "Geom2VecVAMPProjector",
    "Geom2VecTopologyCache",
    "TorchGeom2VecTopologyCache",
    "TrainedVAMPLobe",
    "train_vamp_lobe_from_cache",
]
