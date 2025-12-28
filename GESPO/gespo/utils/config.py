"""Helper for summarising and logging CLI configurations."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from .._imports import argparse, np, torch

logger = logging.getLogger(__name__)

class RunConfigurer:
    """Tracks parsed CLI arguments for printing and optional JSON logging."""

    def __init__(self, args: argparse.Namespace, log_dir: Optional[str] = None) -> None:
        self.args = args
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir else None

    def as_dict(self) -> dict:
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
        logger.info("=== Configuration Summary ===")
        for line in self.formatted_lines():
            logger.info(line)
        logger.info("=============================")

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
