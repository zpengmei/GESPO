"""Programmatic API for GESPO training workflows.

These are the ``pretrain`` and ``train`` stages of the pipeline. The preceding
``seed`` stage is ``gespo.generate_seeds`` (see ``gespo.seed``). A full run::

    import gespo

    # 1. seed: short biased MD -> seeds/walker_*/production.dcd
    gespo.generate_seeds(method="metad", pdb="protein.pdb", out_dir="seeds",
                         cv_set="folding", n_walkers=4)

    # 2. pretrain the TorsionAR policy on the seeds
    gespo.pretrain_torsion_ar(source_path="seeds/", out_path="checkpoints/r0.pt",
                              pdb_path="protein.pdb", global_token_set="rmsd_e2e_rg")

    # 3. train (from a YAML path, a dict, or a TrainingConfig)
    from gespo.config import load_config
    cfg = load_config("config.yaml")
    cfg.episodes = 10
    gespo.train(cfg)

See ``docs/workflow.md`` for the end-to-end walkthrough and the multi-round protocol.
"""
from __future__ import annotations

from typing import Optional


def train(config) -> None:
    """Run GESPO training with the given configuration.

    Parameters
    ----------
    config : TrainingConfig or dict or str
        - ``TrainingConfig``: use directly.
        - ``dict``: passed to ``TrainingConfig.from_dict()``.
        - ``str``: treated as a YAML file path and loaded via ``load_config()``.
    """
    from .config import TrainingConfig, load_config
    from .training import train as _train
    import argparse

    if isinstance(config, str):
        config = load_config(config)
    elif isinstance(config, dict):
        config = TrainingConfig.from_dict(config)

    # Convert to argparse.Namespace for backward compatibility with training loops.
    args = argparse.Namespace(**config.to_dict())
    _train(args)


def pretrain_torsion_ar(
    source_path: str,
    out_path: str,
    pdb_path: str,
    *,
    pdb_b_path: Optional[str] = None,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[str] = None,
    stride: int = 1,
    global_token_set: str = "rmsd_e2e",
    torsion_mode: str = "backbone",
    ligand_smiles: str = "",
    reweight_zenodo_dir: str = "",
    reweight_ref_pdb: str = "",
    reweight_labels_npz: str = "",
    reweight_alpha: float = 1.0,
    pocket_ref_pdb: str = "",
) -> None:
    """Pretrain a TorsionAR policy from trajectory data.

    Parameters
    ----------
    source_path : str
        Path to DCD/trajectory directory or file.
    out_path : str
        Output path for the pretrained checkpoint.
    pdb_path : str
        PDB file for topology.
    global_token_set : str, optional
        Global CV tokens the policy conditions on (default ``"rmsd_e2e"``).
        Validated against the known set — an unknown name raises with the valid
        list. See ``docs/reference.md`` for the catalog.
    torsion_mode : str, optional
        ``"backbone"`` (default) or ``"backbone+sidechain"`` (adds chi1-chi4).
        The legacy spelling ``"backbone_chi"`` is accepted and mapped.
    ligand_smiles : str, optional
        SMILES of the ligand (enables RDKit-based rotatable-bond detection).
    reweight_zenodo_dir, reweight_ref_pdb, reweight_labels_npz, reweight_alpha : optional
        Macrostate/cluster importance-reweighting controls (see the CLI
        ``gespo pretrain --help`` for details).
    pocket_ref_pdb : str, optional
        Reference PDB with a bound ligand for pocket auto-detection when
        ``pdb_path`` is an apo structure.

    Notes
    -----
    This is a faithful superset of the ``gespo pretrain`` CLI: every parameter
    the CLI forwards is accepted here with identical defaults. Model-architecture
    knobs (``d_model``, ``n_layers``, ``n_heads``, ``kmix``) are left at their
    library defaults; call ``gespo.training.pretrain_torsion_ar`` directly to set
    them.
    """
    from .training.pretrain_torsion_ar import pretrain_torsion_ar as _pretrain

    _pretrain(
        source_path=source_path,
        out_path=out_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
        pdb_path=pdb_path,
        pdb_b_path=pdb_b_path,
        stride=stride,
        global_token_set=global_token_set,
        torsion_mode=torsion_mode,
        ligand_smiles=ligand_smiles,
        reweight_zenodo_dir=reweight_zenodo_dir,
        reweight_ref_pdb=reweight_ref_pdb,
        reweight_labels_npz=reweight_labels_npz,
        reweight_alpha=reweight_alpha,
        pocket_ref_pdb=pocket_ref_pdb,
    )
