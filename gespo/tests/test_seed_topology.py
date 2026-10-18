"""The DCD-to-pretraining handoff must retain its exact solvent topology."""
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip('openmm')
from openmm import app, unit
import mdtraj as md

from gespo.seed.system import write_simulation_topology
from gespo.training.pretrain_torsion_ar import _build_torsion_dataset_from_dcd


def test_full_system_seed_topology_allows_pretraining(tmp_path):
    source = Path(app.__file__).parent / 'data/test.pdb'
    reference = md.load(str(source))
    # Change the solvent atom count to ensure the original input topology cannot
    # be used to interpret the full-system DCD.
    protein = reference.topology.select('protein')
    waters = reference.topology.select('water')[:3]
    seed = reference.atom_slice(np.concatenate([protein, waters]))
    topology = seed.topology.to_openmm()
    positions = seed.xyz[0] * unit.nanometer
    write_simulation_topology(tmp_path, topology, positions)
    seed.save_dcd(str(tmp_path / 'production.dcd'))
    torsions, mask, residues, globals_ = _build_torsion_dataset_from_dcd(
        str(tmp_path), str(source), global_token_set='rmsd_e2e')
    assert torsions.shape[0] == 1
    assert mask.any()
    assert np.isfinite(torsions).all() and np.isfinite(globals_).all()
    loaded = md.load(str(tmp_path / 'production.dcd'),
                     top=str(tmp_path / 'simulation_topology.pdb'))
    np.testing.assert_allclose(loaded.xyz, seed.xyz, atol=1e-6)


def test_saved_topology_preserves_atom_names_and_box(tmp_path):
    source = app.PDBFile(str(Path(app.__file__).parent / 'data/test.pdb'))
    box = source.topology.getPeriodicBoxVectors()
    path = write_simulation_topology(tmp_path, source.topology, source.positions, box)
    restored = app.PDBFile(str(path))
    assert [a.name for a in restored.topology.atoms()] == [a.name for a in source.topology.atoms()]
    np.testing.assert_allclose(restored.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer),
                               box.value_in_unit(unit.nanometer), atol=1e-4)


def test_high_temperature_restart_reuses_saved_topology(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from gespo.seed.methods import high_temp

    source = app.PDBFile(str(Path(app.__file__).parent / 'data/test.pdb'))
    write_simulation_topology(tmp_path, source.topology, source.positions)
    (tmp_path / 'production.chk').write_bytes(b'checkpoint-test-double')
    simulation = MagicMock()
    monkeypatch.setattr(high_temp._sys, 'make_forcefield', lambda **kwargs: object())
    monkeypatch.setattr(high_temp._sys, 'prepare_topology',
                        MagicMock(side_effect=AssertionError('Restart must not re-solvate.')))
    monkeypatch.setattr(high_temp._sys, 'platform_and_props', lambda *args, **kwargs: (None, {}))
    monkeypatch.setattr(high_temp._sys, 'build_system', lambda *args, **kwargs: object())
    def make_sim(topology, *args, **kwargs):
        assert topology.getNumAtoms() == source.topology.getNumAtoms()
        return simulation
    monkeypatch.setattr(high_temp._sys, 'make_sim', make_sim)
    monkeypatch.setattr(high_temp, '_attach_reporters', lambda *args, **kwargs: None)
    high_temp.run_high_temp(pdb='original-input-is-not-needed.pdb', out_dir=str(tmp_path),
                            temperature=400, restart=True, production_ns=.000002)
    simulation.loadCheckpoint.assert_called_once_with(str(tmp_path / 'production.chk'))
    simulation.step.assert_called_once_with(1)
