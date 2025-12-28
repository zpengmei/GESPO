"""Helper for summarising and logging CLI configurations."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from .._imports import argparse, np, torch
from ..config import TrainingConfig

logger = logging.getLogger(__name__)

class RunConfigurer:
    """Tracks parsed CLI arguments for printing and optional JSON logging."""

    def __init__(self, args: argparse.Namespace, log_dir: Optional[str] = None) -> None:
        self.args = args
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir else None

    def as_dict(self) -> dict:
        if isinstance(self.args, TrainingConfig) or is_dataclass(self.args):
            return self.args.to_dict() if hasattr(self.args, "to_dict") else asdict(self.args)
        try:
            config = TrainingConfig.from_args(self.args)
            return config.to_dict()
        except Exception:
            return {key: getattr(self.args, key) for key in sorted(vars(self.args))}

    def formatted_lines(self) -> list[str]:
        config_dict = self.as_dict()
        if not isinstance(config_dict, dict):
            return [str(config_dict)]

        lines = self._format_config(config_dict)
        lines.extend(self._format_metadata())
        return lines

    def _format_config(self, config_dict: dict) -> list[str]:
        if not any(isinstance(value, dict) for value in config_dict.values()):
            return [f"{key:24s}: {config_dict[key]}" for key in sorted(config_dict)]

        lines: list[str] = []
        for section, values in config_dict.items():
            if isinstance(values, dict):
                lines.append(f"[{section}]")
                for key, value in values.items():
                    lines.append(f"  {key:22s}: {value}")
            else:
                lines.append(f"{section:24s}: {values}")
        return lines

    def _format_metadata(self) -> list[str]:
        extra = [*self._extract_ckpt_metadata(), *self._extract_dataset_metadata()]
        return [f"{key:24s}: {value}" for key, value in extra]

    def _extract_ckpt_metadata(self) -> list[tuple[str, object]]:
        ckpt_path = getattr(self.args, "cond_checkpoint", None)
        if not ckpt_path:
            return []

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt.get("config", {}) or {}
        except Exception:
            return []

        keys = (
            "conditioning_mode",
            "cond_injection_mode",
            "atom_mixer",
            "num_residue_sinks",
            "hybrid_torsion_dim",
            "hybrid_pair_dim",
            "diffusion_steps",
        )
        return [(f"ckpt.{key}", cfg[key]) for key in keys if key in cfg]

    def _extract_dataset_metadata(self) -> list[tuple[str, object]]:
        data_path = getattr(self.args, "cond_data", None)
        if not data_path:
            return []

        try:
            data = np.load(data_path)
        except Exception:
            return []

        def get_arr(name: str) -> Optional[object]:
            if name not in data.files:
                return None
            val = data.get(name)
            try:
                return val.item() if hasattr(val, "item") else int(val)
            except Exception:
                return val

        values = [
            ("data.conditioning_mode", get_arr("conditioning_mode")),
            ("data.hybrid_torsion_dim", get_arr("hybrid_torsion_dim")),
            ("data.hybrid_pair_dim", get_arr("hybrid_pair_dim")),
        ]
        return [(key, value) for key, value in values if value is not None]

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
