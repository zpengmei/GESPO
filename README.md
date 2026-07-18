# GESPO (Generative Enhanced Sampling for Peptide Optimization)

RL-based enhanced sampling toolkit with a geom2vec VAMP backbone for protein
conformational sampling.

## Layout

```
gespo/
├── __init__.py             # Curated public API (lazy): train, pretrain_torsion_ar,
│                           #   TrainingConfig, load_config, save_config, generate_template
├── __main__.py             # `python -m gespo` → cli.main
├── api.py                  # Programmatic entry points (train, pretrain_torsion_ar)
├── cli.py                  # CLI entry point (train / pretrain / config / bootstrap)
├── bootstrap_cli.py        # RMSD-biased bootstrap MD utilities (python -m gespo.bootstrap_cli)
├── config/                 # Configuration package (dataclasses + argparse + YAML)
│   ├── training.py         #   TrainingConfig (root) + sub-configs below
│   ├── md.py, io.py, geom2vec.py, committor.py, steering.py,
│   │   path_sampling.py, msm_ig.py, vamp_bin.py
│   ├── cli_gen.py          #   Auto-generate argparse flags from dataclass fields
│   └── loader.py           #   YAML load/save + templates
├── loops/                  # Training loops
│   ├── base.py             #   BaseTrainingLoop (template-method epoch structure)
│   ├── generative.py       #   TorsionAR generative loop
│   ├── generative_path.py  #   Path-sampling loop (committor rewards)
│   ├── _helpers.py         #   Shared loop helpers
│   └── _legacy/            #   discrete-start + toy phi/psi loops
├── policies/               # GRPOAgent, StateLogitPolicy, VonMisesStartPolicy, TorsionARPolicy
├── rewards/                # RewardCombiner + LOO-VAMP / VAMP-bin / MSM-IG / committor / frontier
├── projections/            # VAMP projectors + geom2vec/ embedding backbone
│   └── geom2vec/           #   geom2vec embedding cache, inference, VAMP lobe (see its README)
├── sampling/               # OpenMM environment (env.py) + replay steering (targeted_md.py)
├── training/               # RL agent, GRPO optimizer, sampler, checkpointing, TorsionAR
│   │                       #   pretraining/eval, and components/ (extracted loop pieces)
│   └── components/         #   RetrievalManager, Geom2VecManager, CommittorState, anchoring, …
├── analysis/               # Diagnostics, CV metrics, plotting
├── models/                 # Transformer blocks + positional encodings
├── utils/                  # Geometry, OpenMM helpers, endpoints, selection, constants
└── runner_agent/           # LLM-powered experiment runner (separate `gespo-agent` CLI)
```

## How to run (CLI)

Start from a generated config template, then train:

```bash
python -m gespo config template --mode generative > my_run.yaml
# edit my_run.yaml: set endpoints.pdb_a to your (hydrogen-added) PDB,
# and geom2vec.checkpoint to your geom2vec representation checkpoint
python -m gespo train --config my_run.yaml
```

CLI flags override YAML, and any flag can be passed directly:

```bash
python -m gespo train \
    --pdb your_protein.pdb \
    --generative-loop \
    --projection geom2vec_vamp \
    --geom2vec-ckpt /path/to/geom2vec.pt
```

Other subcommands:

```bash
python -m gespo pretrain --pdb a.pdb --source dcd_dir/ --out ckpt.pt  # TorsionAR pretraining
python -m gespo config validate my_run.yaml                          # validate a config
python -m gespo.bootstrap_cli --help                                 # RMSD-biased bootstrap MD
```

There is no bundled example structure — bring your own PDB (see
`add_hydrogens_pdbfixer.py` for a hydrogen-adding helper). A minimal starting
config lives at `examples/generative_min.yaml`.

## Programmatic use

```python
import gespo

# Train from a YAML path, a dict, or a TrainingConfig
gespo.train("my_run.yaml")

# Inspect / modify configuration first
cfg = gespo.load_config("my_run.yaml")
cfg.episodes = 10
cfg.snapshot_dir = "runs/round1"
gespo.train(cfg)

# Pretrain a TorsionAR policy from trajectory data
gespo.pretrain_torsion_ar(
    source_path="history_dcds/",
    out_path="checkpoints/r0.pt",
    pdb_path="endpoint_A.pdb",
)
```

`import gespo` is cheap — the scientific stack (OpenMM/torch) is only imported
when you actually call one of these entry points.

## Notes

- The geom2vec VAMP backbone has its own guide: `gespo/projections/geom2vec/README.md`.
- Configuration is a package under `gespo/config/`; CLI flags are auto-generated
  from the dataclass fields (`config/cli_gen.py`), and configs round-trip to YAML
  (`config/loader.py`).
- End-of-epoch CV diagnostics include PCA coverage and Vendi novelty.
- The sampling environment and replay steering live under `gespo/sampling/`;
  RL code (GRPO agent + loops) lives under `gespo/policies/`, `gespo/training/`,
  and `gespo/loops/`.
