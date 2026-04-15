import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import torch

# Mock optional heavy deps so `gespo` import doesn't require OpenMM in this environment.
_openmm = MagicMock()
sys.modules["openmm"] = _openmm
sys.modules["openmm.app"] = MagicMock()
_openmm_unit = MagicMock()
_openmm_unit.Quantity = type("Quantity", (), {})
sys.modules["openmm.unit"] = _openmm_unit
_openmm.unit = _openmm_unit

sys.path.append("/project/dinner/zpengmei/GESPO")

import mdtraj as md

from gespo.training.geom2vec_backbone import EmbeddingCache, Geom2VecTopologyCache, infer_and_cache, infer_geom2vec_embeddings


class DummyGeom2VecRep(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.lin = torch.nn.Linear(3, self.hidden_channels, bias=False)

    def forward(self, z, pos, batch):
        # pos: (n_atoms_total, 3)
        x = self.lin(pos)  # (n_atoms_total, H)
        v = x.unsqueeze(1).expand(-1, 3, -1)  # (n_atoms_total, 3, H)
        return x, v, None


def _build_two_residue_topology() -> md.Topology:
    top = md.Topology()
    chain = top.add_chain()
    res0 = top.add_residue("ALA", chain)
    res1 = top.add_residue("GLY", chain)

    # Residue 0 atoms
    top.add_atom("N", md.element.nitrogen, res0)
    top.add_atom("CA", md.element.carbon, res0)
    top.add_atom("C", md.element.carbon, res0)
    top.add_atom("O", md.element.oxygen, res0)
    top.add_atom("H", md.element.hydrogen, res0)

    # Residue 1 atoms
    top.add_atom("N", md.element.nitrogen, res1)
    top.add_atom("CA", md.element.carbon, res1)
    top.add_atom("C", md.element.carbon, res1)
    top.add_atom("O", md.element.oxygen, res1)
    top.add_atom("H", md.element.hydrogen, res1)

    return top


def test_topology_cache_and_inference_shapes():
    top = _build_two_residue_topology()
    cache = Geom2VecTopologyCache.from_mdtraj(top, exclude_hydrogens=True)

    assert cache.num_tokens == 2
    assert cache.num_atoms == 8  # 10 atoms total - 2 hydrogens
    assert cache.cg_counts.tolist() == [4, 4]
    assert cache.ca_indices.tolist() == [1, 5]

    coords_nm = np.random.RandomState(0).randn(5, top.n_atoms, 3).astype(np.float32) * 0.1
    model = DummyGeom2VecRep(hidden_channels=6)

    graph, ca = infer_geom2vec_embeddings(
        model=model,
        topology=cache,
        coords=coords_nm,
        coords_in_nm=True,
        stride=2,
        batch_size=2,
        reduction="sum",
        hidden_channels=6,
        device="cpu",
        center_ca=True,
    )

    assert isinstance(graph, torch.Tensor) and isinstance(ca, torch.Tensor)
    assert graph.dtype == torch.float32 and ca.dtype == torch.float32
    assert graph.shape == (3, 2, 4, 6)  # frames 0,2,4
    assert ca.shape == (3, 2, 3)
    assert torch.allclose(ca.mean(dim=1), torch.zeros((3, 3), dtype=torch.float32), atol=1e-5)


def test_pipeline_infer_and_cache_roundtrip():
    top = _build_two_residue_topology()
    topo_cache = Geom2VecTopologyCache.from_mdtraj(top, exclude_hydrogens=True)
    coords_nm = np.random.RandomState(1).randn(4, top.n_atoms, 3).astype(np.float32) * 0.05
    model = DummyGeom2VecRep(hidden_channels=4)

    with tempfile.TemporaryDirectory() as tmp:
        emb_cache = EmbeddingCache(tmp)
        rec = infer_and_cache(
            emb_cache,
            key="epoch_0001/traj_00",
            model=model,
            topology=topo_cache,
            coords=coords_nm,
            stride=1,
            batch_size=2,
            hidden_channels=4,
            device="cpu",
            overwrite=True,
            meta={"unit_test": True},
        )
        assert rec.dtype == "float32"

        graph, ca = emb_cache.load("epoch_0001/traj_00")
        assert graph.dtype == torch.float32 and ca.dtype == torch.float32
        assert graph.shape == (4, 2, 4, 4)
        assert ca.shape == (4, 2, 3)


if __name__ == "__main__":
    test_topology_cache_and_inference_shapes()
    test_pipeline_infer_and_cache_roundtrip()
    print("✓ geom2vec_backbone topology/inference tests passed")
