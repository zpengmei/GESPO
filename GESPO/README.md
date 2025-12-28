# GESPO (Generative Enhanced Sampling for Peptide Optimization)

This repository refactors the original monolithic `chig_gen.py` script into a structured Python package with clearly defined subpackages for training, sampling, utilities, diffusion models, and CLI handling.

## Layout

```
gespo/
├── __init__.py
├── __main__.py
├── _imports.py              # heavy third-party imports and sys.path tweaks
├── cli.py                   # CLI entry point
├── diffusion/               # diffusion schedule utilities (unchanged)
├── models/                  # neural network building blocks
├── sampling/
│   ├── __init__.py
│   ├── env.py               # OpenMM environment + CV tracking
│   └── generator.py         # conditional diffusion wrapper
├── training/
│   ├── __init__.py
│   ├── loop.py              # GRPO + MD training loop
│   ├── rl.py                # actor policy definition
│   └── scorer.py            # PCA scorer + Vendi novelty
└── utils/
    ├── __init__.py
    ├── constants.py         # metadynamics/force-group constants
    ├── geometry.py          # torsion utilities, RMSD, Kabsch, etc.
    ├── openmm_utils.py      # device selection + Gaussian bias helper
    └── plotting.py          # Ramachandran + analysis plots
```

## How to run

```bash
# from the repo root
python -m gespo --help
python -m gespo \
    --pdb examples/chignolin/chignolin.pdb \
    --cond-checkpoint examples/chignolin/diffusion_cartesian_conditional_93atoms_32latent_split.pth \
    --cond-data examples/chignolin/chignolin_data_aligned.npz

# end-to-end diffusion data prep + training
python -m gespo.diffusion_cli \
    --pdb examples/chignolin/chignolin.pdb \
    --traj-dir /path/to/dcd_folder \
    --work-dir diffusion_runs/chignolin
```

Or import programmatically:

```python
from gespo import train
args = build_args_somehow()
train(args)
```

## Notes

- Utilities shared across training and sampling live under `gespo/utils/`.
- `gespo/utils/config.py` provides the `RunConfigurer` used by the CLI to print and optionally log parsed arguments (enable JSON dumps via `--config-log path/to/dir`).
- End-of-epoch CV diagnostics include PCA coverage and Vendi novelty.
- `python -m gespo.diffusion_cli` automates dataset preparation (PDB + trajectory folder → NPZ) and diffusion training (producing a checkpoint compatible with `gespo.sampling.generator`).
  - Supports torsion, sparse pairwise-distance, or hybrid conditioning (`--conditioning-mode torsion|pairwise|hybrid`).
  - Choose how conditioning feeds into the diffusion network (`--cond-injection-mode global|per_residue`).
  - Provide `--config config.yaml` to override defaults (data/model/training) via YAML.
- When sampling diffusion checkpoints trained with pairwise/hybrid conditioning, pass the corresponding pairwise feature vectors to `ConditionalGenerator.sample()/sample_batch` via the `condition_vector` argument.
- The conditional generator and sampling environment live under `gespo/sampling/`.
- Reinforcement-learning code (GRPO agent + training loop) lives under `gespo/training/`.
- Diffusion schedules remain in `gespo/diffusion/`; neural network modules live in `gespo/models/`.

If you want stricter imports (avoid `from ._imports import *`), we can make a follow-up pass that inlines only the necessary imports per module.
