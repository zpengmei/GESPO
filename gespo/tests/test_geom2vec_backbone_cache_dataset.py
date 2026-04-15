import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

# Mock optional heavy deps (mirrors existing tests)
_openmm = MagicMock()
sys.modules["openmm"] = _openmm
sys.modules["openmm.app"] = MagicMock()
_openmm_unit = MagicMock()
_openmm_unit.Quantity = type("Quantity", (), {})
sys.modules["openmm.unit"] = _openmm_unit
_openmm.unit = _openmm_unit

# Add project root to path (mirrors existing tests)
sys.path.append("/project/dinner/zpengmei/GESPO")

from gespo.training.geom2vec_backbone.cache import EmbeddingCache
from gespo.training.geom2vec_backbone.dataset import PairIndex, TimeLaggedIndexDataset


def test_embedding_cache_stores_float32():
    with tempfile.TemporaryDirectory() as tmp:
        cache = EmbeddingCache(tmp)

        graph = torch.zeros((5, 2, 4, 3), dtype=torch.float16)
        ca = torch.zeros((5, 2, 3), dtype=torch.float32)

        rec = cache.store("epoch_0001/traj_00", graph_features=graph, ca_coords=ca)
        assert rec.dtype == "float32"
        assert Path(rec.embedding_path).exists()
        assert Path(rec.ca_path).exists()

        graph_loaded, ca_loaded = cache.load("epoch_0001/traj_00")
        assert graph_loaded.dtype == torch.float32
        assert ca_loaded.dtype == torch.float32
        assert graph_loaded.shape[0] == 5
        assert ca_loaded.shape == (5, 2, 3)


def test_time_lagged_index_dataset_basic_mapping():
    g0 = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)
    g1 = torch.arange(7 * 3, dtype=torch.float32).reshape(7, 3) + 100
    trajectories = [{"graph_features": g0}, {"graph_features": g1}]

    ds = TimeLaggedIndexDataset(trajectories, lag_time=2, pair_stride=1, return_index=True)
    # pairs: (5-2) + (7-2) = 8
    assert len(ds) == 8

    inst, lag = ds[0]
    assert torch.allclose(inst["graph_features"], g0[0])
    assert torch.allclose(lag["graph_features"], g0[2])
    assert isinstance(inst["index"], PairIndex)
    assert inst["index"] == PairIndex(traj_idx=0, t=0, t_lag=2)

    inst, lag = ds[2]
    assert torch.allclose(inst["graph_features"], g0[2])
    assert torch.allclose(lag["graph_features"], g0[4])

    inst, lag = ds[3]
    # first pair from trajectory 1
    assert torch.allclose(inst["graph_features"], g1[0])
    assert torch.allclose(lag["graph_features"], g1[2])


def test_time_lagged_index_dataset_pair_stride():
    g = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)
    ds = TimeLaggedIndexDataset([g], lag_time=3, pair_stride=2)
    # valid t: 0,2,4,6  (t+3 < 10)
    assert len(ds) == 4

    inst, lag = ds[1]
    assert torch.allclose(inst["graph_features"], g[2])
    assert torch.allclose(lag["graph_features"], g[5])


if __name__ == "__main__":
    test_embedding_cache_stores_float32()
    test_time_lagged_index_dataset_basic_mapping()
    test_time_lagged_index_dataset_pair_stride()
    print("✓ geom2vec_backbone cache/dataset tests passed")
