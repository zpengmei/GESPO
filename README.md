# GESPO — Generative Enhanced Sampling by Policy Optimization

GESPO is a **self-improving agentic system for biomolecular kinetics discovery
from molecular dynamics**. It post-trains a pretrained generative *launch policy*
with policy optimization (GRPO) to decide **where new MD trajectories should
start**, so that short unbiased simulations expose the slow processes — metastable
states, transition pathways, committors — that static structure/ensemble
generators miss. Crucially, GESPO changes only the launch distribution; it never
alters the physical dynamics used for production.

> Method paper: *Recursive agentic post-training for biomolecular kinetics
> discovery from molecular dynamics* — Pengmei, Nguyen, Shen, Shu, Weare & Dinner.
> See [Citation](#citation).

## What GESPO does

Each round is an agentic loop that separates the **decision layer** (which MD
experiments to run) from the **propagation layer** (unbiased physics):

1. **Sample launch descriptions** from the policy πθ — a compact, autoregressive
   description of a starting structure: global descriptors (RMSD-like distances),
   protein backbone torsions (φ, ψ), and, for protein–ligand systems, ligand
   rotatable-bond torsions. Von Mises heads for torsions, Gaussian mixtures for
   global descriptors give an exact, normalized launch density (the **TorsionAR**
   policy).
2. **Realize** each description into an all-atom configuration via a
   **retrieval-augmented replay-and-steer map**: pick a nearby frame from the
   trajectory history, briefly steer it toward the proposed torsions/descriptors,
   then release — steering is absent during the production segment.
3. **Propagate** N unbiased short MD segments in parallel (the environment: an
   OpenMM engine with your force field and thermodynamic conditions).
4. **Score** each segment by its **kinetic contribution** — its leave-one-out
   ΔVAMP-2 gain in a learned slow-dynamical (geom2vec-VAMP) representation
   (committor-based rewards for transition-path systems).
5. **Update** πθ with GRPO, regularized toward the pretrained generator.

Then GESPO **recurses**: accumulate the new trajectories, re-pretrain the policy,
and iterate — exploiting the growing MD corpus to map the kinetic landscape with
orders of magnitude less simulation time than direct MD.

## The pipeline (how the loop maps to commands)

```
  gespo seed  ──▶  gespo pretrain  ──▶  gespo train  ──▶  iterate rounds
  initial MD        the launch policy   the agentic loop above
  (the corpus)      (max-likelihood     (realize → propagate →
                     prior)              ΔVAMP-2 reward → GRPO)
```

New here? Read [`docs/overview.md`](docs/overview.md), then run a recipe in
[`examples/`](examples/). Full docs: [`docs/`](docs/).

## Quickstart

You provide a **prepared PDB** and a **geom2vec backbone checkpoint** (see
[`docs/inputs.md`](docs/inputs.md)); `gespo seed` produces the initial corpus.

```bash
# 1. seed: short biased MD -> seeds/walker_*/production.dcd
python -m gespo seed metad --pdb protein.pdb --cv-set folding \
    --implicit-solvent --n-walkers 4 --out-dir seeds

# 2. pretrain the launch policy (max-likelihood prior) on the seeds
python -m gespo pretrain --pdb protein.pdb --source seeds/walker_1,seeds/walker_2,seeds/walker_3,seeds/walker_4 \
    --out torsion_ar.pt --global-token-set rmsd_e2e_rg

# 3. train: the GRPO post-training loop
python -m gespo train --pdb protein.pdb --generative-loop --implicit-solvent \
    --torsion-ar-checkpoint torsion_ar.pt \
    --projection geom2vec_vamp --geom2vec-ckpt /path/to/backbone.pth \
    --replay-only-dcd-dir seeds/walker_1,seeds/walker_2,seeds/walker_3,seeds/walker_4 \
    --snapshot-dir runs/round1
```

Worked, copy-pasteable recipes for the paper's system classes — peptide folding
and a protein–ligand pocket:
[`examples/protein_folding_trpcage/`](examples/protein_folding_trpcage/) and
[`examples/protein_ligand_trypsin/`](examples/protein_ligand_trypsin/).

### Config files

Any run can be driven by a YAML `--config` instead of long flag lists (CLI flags
override YAML). Generate a starting template, or use an example:

```bash
python -m gespo config template --mode generative > my_run.yaml
python -m gespo train --config my_run.yaml
python -m gespo config validate my_run.yaml --strict   # check for typos
```

`gespo train --help` shows the essential flags; `--help-all` shows all of them.

## Programmatic use

```python
import gespo

# Generate the corpus, pretrain the policy, then post-train
gespo.generate_seeds(method="metad", pdb="protein.pdb", out_dir="seeds",
                     cv_set="folding", n_walkers=4)
gespo.pretrain_torsion_ar(source_path="seeds/", out_path="torsion_ar.pt",
                          pdb_path="protein.pdb", global_token_set="rmsd_e2e_rg")
gespo.train("my_run.yaml")   # a YAML path, a dict, or a TrainingConfig

cfg = gespo.load_config("my_run.yaml")
cfg.episodes = 10
gespo.train(cfg)
```

`import gespo` is cheap — OpenMM/torch load only when you call one of these.

## Layout

```
gespo/
├── __init__.py             # Curated public API (lazy): train, pretrain_torsion_ar,
│                           #   generate_seeds, TrainingConfig, load/save_config, generate_template
├── __main__.py             # `python -m gespo` → cli.main
├── api.py                  # Programmatic entry points (train, pretrain_torsion_ar)
├── cli.py                  # CLI entry point (seed / pretrain / train / config / bootstrap)
├── seed/                   # Seed generation: `gespo seed {metad,high-temp,rmsd-bias}`
│   ├── system.py, restraints.py, cvs.py   #   shared OpenMM system-building core + CV registry
│   ├── methods/            #   metad / high_temp / rmsd_bias
│   └── walkers.py, cli.py  #   multi-walker fan-out + subparser
├── bootstrap_cli.py        # RMSD-biased bootstrap MD (also `gespo seed rmsd-bias`)
├── config/                 # Configuration package (dataclasses + argparse + YAML)
│   ├── training.py         #   TrainingConfig (root) + sub-configs below
│   ├── md.py, io.py, geom2vec.py, committor.py, steering.py,
│   │   path_sampling.py, msm_ig.py, vamp_bin.py
│   ├── cli_gen.py          #   Auto-generate argparse flags from dataclass fields
│   └── loader.py           #   YAML load/save + templates
├── training/               # RL agent, GRPO optimizer, sampler, checkpointing,
│   │                       #   TorsionAR launch-policy pretraining (+ token_sets.py), components/
│   └── components/         #   RetrievalManager, Geom2VecManager, CommittorState, anchoring, …
├── loops/                  # Post-training loops
│   ├── base.py             #   BaseTrainingLoop (template-method epoch structure)
│   ├── generative.py       #   TorsionAR generative loop
│   ├── generative_path.py  #   Path-sampling loop (committor rewards)
│   ├── _helpers.py         #   Shared loop helpers
│   └── _legacy/            #   discrete-start + toy phi/psi loops
├── policies/               # GRPOAgent, StateLogitPolicy, VonMisesStartPolicy, TorsionARPolicy
├── rewards/                # RewardCombiner + LOO-ΔVAMP-2 / VAMP-bin / MSM-IG / committor / frontier
├── projections/            # VAMP projectors + geom2vec/ embedding backbone
│   └── geom2vec/           #   geom2vec embedding cache, inference, VAMP lobe (see its README)
├── sampling/               # OpenMM environment (env.py) + replay-and-steer map (targeted_md.py)
├── analysis/               # Diagnostics, CV metrics, plotting
├── models/                 # Transformer blocks + positional encodings
├── utils/                  # Geometry, OpenMM helpers, endpoints, selection, log, constants
└── runner_agent/           # LLM-powered experiment runner (separate `gespo-agent` CLI)
```

## Documentation

- [`docs/overview.md`](docs/overview.md) — what GESPO does; the agentic loop; loop phases
- [`docs/inputs.md`](docs/inputs.md) — the PDB, geom2vec backbone, and seed inputs
- [`docs/seeding.md`](docs/seeding.md) — `gespo seed` methods and CV sets
- [`docs/workflow.md`](docs/workflow.md) — end-to-end walkthrough + recursive multi-round protocol
- [`docs/ligands.md`](docs/ligands.md) — protein–ligand specifics
- [`docs/reference.md`](docs/reference.md) — token sets, rewards, flags, YAML sections
- [`gespo/projections/geom2vec/README.md`](gespo/projections/geom2vec/README.md) — the geom2vec VAMP backbone

## Citation

If you use GESPO, please cite:

> Zihan Pengmei, Nhan D. Nguyen, Zhengyuan Shen, Yinan Shu, Jonathan Weare, and
> Aaron R. Dinner. *Recursive agentic post-training for biomolecular kinetics
> discovery from molecular dynamics* (2026).
