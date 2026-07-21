"""Helper for summarising and logging CLI configurations."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import argparse

import numpy as np
import torch


class RunConfigurer:
    """Tracks parsed CLI arguments for printing and optional JSON logging."""

    def __init__(self, args: argparse.Namespace, log_dir: Optional[str] = None) -> None:
        self.args = args
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir else None

    def as_dict(self) -> dict:
        # Resolve the effective configuration so the summary/dump show real
        # values rather than the ``None`` unset-sentinel that unspecified CLI
        # flags now carry. Falls back to the raw namespace if resolution fails.
        try:
            from gespo.config import TrainingConfig
            return TrainingConfig.from_args(self.args).to_dict()
        except Exception:
            return {key: getattr(self.args, key) for key in sorted(vars(self.args))}

    def formatted_lines(self) -> list[str]:
        config_dict = self.as_dict()
        lines = [f"{key:24s}: {config_dict[key]}" for key in config_dict]
        # Append generator/dataset metadata if available for clarity
        extra = []
        try:
            ckpt_path = getattr(self.args, "cond_checkpoint", None)
            if ckpt_path:
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    cfg = ckpt.get("config", {}) or {}
                    def add_ck(key):
                        if key in cfg:
                            extra.append((f"ckpt.{key}", cfg[key]))
                    for k in (
                        "conditioning_mode",
                        "cond_injection_mode",
                        "atom_mixer",
                        "num_residue_sinks",
                        "hybrid_torsion_dim",
                        "hybrid_pair_dim",
                        "diffusion_steps",
                    ):
                        add_ck(k)
                except Exception:
                    pass
            data_path = getattr(self.args, "cond_data", None)
            if data_path:
                try:
                    data = np.load(data_path)
                    def get_arr(name):
                        if name in data.files:
                            val = data.get(name)
                            try:
                                return val.item() if hasattr(val, "item") else int(val)
                            except Exception:
                                return val
                        return None
                    cmode = get_arr("conditioning_mode")
                    if cmode is not None:
                        extra.append(("data.conditioning_mode", cmode))
                    ht = get_arr("hybrid_torsion_dim")
                    if ht is not None:
                        extra.append(("data.hybrid_torsion_dim", ht))
                    hp = get_arr("hybrid_pair_dim")
                    if hp is not None:
                        extra.append(("data.hybrid_pair_dim", hp))
                except Exception:
                    pass
        except Exception:
            pass
        for k, v in extra:
            lines.append(f"{k:24s}: {v}")
        return lines

    def print_summary(self) -> None:
        config_dict = self.as_dict()
        
        # Define field groupings
        groups = {
            "Endpoints": ["pdb", "pdb_b", "path_sampling"],
            "Core Training": ["episodes", "grp_size", "num_md_steps", "lr", "temperature", "output_dir"],
            "Policy": ["torsion_ar_layers", "torsion_ar_heads", "torsion_ar_dim", "torsion_entropy_coef", "ent_coef"],
            "VAMP / Geom2Vec": ["geom2vec_checkpoint", "geom2vec_dim", "geom2vec_lagtime", "geom2vec_stride", 
                                "loo_vamp_weight", "vamp_bin_entropy_coef", "msm_ig_coef"],
            "Retrieval": ["retrieval_k", "retrieval_softmax_temp", "retrieval_max_samples"],
            "Generator / Diffusion": ["cond_checkpoint", "cond_data", "diffusion_steps"],
            "Simulation": ["equilibration_steps", "record_stride", "forcefield"],
        }
        
        print("\n" + "="*60)
        print(" Configuration Summary")
        print("="*60)
        
        printed_keys = set()
        for group_name, keys in groups.items():
            group_items = [(k, config_dict[k]) for k in keys if k in config_dict]
            if group_items:
                print(f"\n[{group_name}]")
                for k, v in group_items:
                    print(f"  {k:28s}: {v}")
                    printed_keys.add(k)
        
        # Print remaining keys not in any group
        remaining = [(k, config_dict[k]) for k in sorted(config_dict) if k not in printed_keys]
        if remaining:
            print(f"\n[Other]")
            for k, v in remaining:
                print(f"  {k:28s}: {v}")
        
        # Print extra metadata from formatted_lines
        extra_lines = [l for l in self.formatted_lines() if l.startswith("ckpt.") or l.startswith("data.")]
        if extra_lines:
            print(f"\n[Checkpoint/Data Metadata]")
            for line in extra_lines:
                print(f"  {line}")
        
        print("\n" + "="*60 + "\n")

    def dump(self, filename: Optional[str] = None) -> Optional[Path]:
        if self.log_dir is None:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"config_{stamp}.json"
        path = self.log_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2, default=str)
        return path
