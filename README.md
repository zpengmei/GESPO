# GESPO (Generative Enhanced Sampling for Peptide Optimization)

RL-based enhanced sampling toolkit with geom2vec VAMP backbone for protein conformational sampling.

## Layout

```
gespo/
├── __init__.py
├── __main__.py
├── cli.py                   # CLI entry point
├── config.py                # Centralized TrainingConfig dataclass + argparse
├── bootstrap_cli.py         # RMSD-biased bootstrap MD utilities
├── models/
│   └── transformer.py       # Transformer blocks and positional encodings
├── sampling/
│   ├── env.py               # OpenMM environment + CV tracking
│   └── targeted_md.py       # Replay steering forces
├── training/
│   ├── __init__.py           # Dispatch to appropriate training loop
│   ├── loop.py               # Standard discrete-start loop
│   ├── loop_generative.py    # Torsion-AR generative loop
│   ├── loop_generative_path.py  # Path-sampling generative loop
│   ├── loop_toy_phi_psi.py   # Toy Ala2-style loop
│   ├── rl.py                 # Policy networks (GRPOAgent, TorsionARPolicy, etc.)
│   ├── sampler.py            # Epoch sampling orchestration
│   ├── optimization.py       # GRPO optimizer
│   ├── vamp_stats.py         # Sufficient statistics for LOO VAMP-2
│   ├── features.py           # Feature builder
│   ├── setup.py              # Policy initialization
│   ├── checkpointing.py      # Checkpoint save/load
│   ├── replay_buffer.py      # Replay buffer
│   ├── dga_committor.py      # DGA committor estimation
│   ├── torsion_retrieval.py   # Torsion nearest-neighbor retrieval
│   ├── pretrain_torsion_ar.py # TorsionAR pretraining
│   ├── eval_torsion_ar.py     # TorsionAR evaluation
│   ├── components/            # Extracted loop components
│   └── geom2vec_backbone/     # Geom2vec embedding + VAMP projector
├── utils/
│   ├── constants.py          # Force-group constants
│   ├── geometry.py           # Torsion utilities, RMSD, Kabsch, etc.
│   ├── openmm_utils.py       # Device selection + Gaussian bias helper
│   ├── plotting.py           # Ramachandran + analysis plots
│   ├── config.py             # RunConfigurer for CLI summary
│   ├── endpoints.py          # Endpoint PDB validation
│   ├── selection.py          # Protein atom selection utilities
│   ├── adk_metrics.py        # ADK-specific CV metrics
│   ├── ppi_metrics.py        # PPI-specific CV metrics
│   └── vinculin_h5_metrics.py
└── runner_agent/              # LLM-powered experiment agent
```

## How to run

```bash
python -m gespo --help
python -m gespo \
    --pdb examples/chignolin/chignolin.pdb \
    --generative-loop \
    --projection geom2vec_vamp \
    --geom2vec-ckpt path/to/geom2vec.pt
```

Or import programmatically:

```python
from gespo.training import train
args = build_args_somehow()
train(args)
```

## Notes

- Utilities shared across training and sampling live under `gespo/utils/`.
- End-of-epoch CV diagnostics include PCA coverage and Vendi novelty.
- The sampling environment and replay steering live under `gespo/sampling/`.
- Reinforcement-learning code (GRPO agent + training loop) lives under `gespo/training/`.
- Neural network modules live in `gespo/models/`.
