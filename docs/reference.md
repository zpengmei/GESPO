# Reference: token sets, rewards, flags, YAML

## Global token sets

`--global-token-set` (pretrain) / `policy.torsion_ar_global_token_set` selects the
global collective-variable scalars the TorsionAR policy conditions on. The common
choices:

| Token set | Global tokens | Use for |
|---|---|---|
| `rmsd_e2e` *(default)* | RMSD-to-A, end-to-end distance | generic single-endpoint |
| `rmsd_e2e_rg` | + radius of gyration | folding / compaction |
| `rmsd_ab_e2e` | RMSD-to-A, RMSD-to-B, e2e | A→B path sampling (needs `--pdb-b`) |
| `rmsd_e2e_pocket3` | RMSD, e2e, pocket RMSD | apo pocket dynamics |
| `rmsd_e2e_pocket_full` | + protein–ligand contacts, ligand↔pocket distance | protein–ligand (holo) |
| `rmsd_e2e_ligdist` | + ligand↔pocket distance | ligand binding/unbinding |

Many aliases and specialized sets (`adk_*`, `vinculin_h5`, `ppi`, `_nm` spellings)
also exist. The authoritative list is `KNOWN_GLOBAL_TOKEN_SETS` in
[`../gespo/training/token_sets.py`](../gespo/training/token_sets.py); an unrecognized
name is now **rejected with an error listing the valid sets** (it used to silently
fall back to `rmsd_e2e`). `--torsion-mode` selects `backbone` (φ/ψ/ω) or
`backbone+sidechain` (adds χ1–χ4).

## Reward terms

`gespo train` combines several diversity rewards (Phase D). Each is off by default
(0 coefficient) unless noted; tune the coefficients to taste.

| Flag | Field | What it rewards |
|---|---|---|
| `--loo-vamp-reward-weight` | `loo_vamp_reward_weight` (default 1.0) | leave-one-out VAMP score gain — the core "expand slow-space" signal |
| `--vamp-bin-entropy-coef` | `vamp_bin_entropy_coef` (−1 → ties to `--ent-coef`) | entropy over VAMP-state occupancy (visit rare states) |
| `--msm-ig-reward-coef` | `msm_ig_reward_coef` | MSM information gain — transitions that most sharpen the estimated kinetic model |
| `--start-distance-reward-coef` | `start_distance_reward_coef` | novelty of the *start* relative to prior starts (+`--pcca-n-macrostates`) |

RL knobs: `--lr`, `--grpo-iters`, `--kl-coef`, `--ent-coef`, `--sync-interval`,
`--grad-clip`, `--grp-size` (walkers/epoch), `--episodes`.

## Path sampling / committor

`--mode path-sampling` (needs `--pdb-b`) samples A→B transition paths. Boundaries:
`--path-boundary-mode` (`rmsd_cutoff` | `vamp_state` | …), `--path-boundary-cutoff-*-nm`.
Committor rewards (all default 0): `--committor-progress-reward-weight`,
`--committor-tube-diversity-reward-weight`, `--committor-connectivity-reward-weight`,
with DGA settings `--committor-dga-n-clusters`, `--committor-dga-lagtime`, etc.
`gespo config template --mode path-sampling` emits a starting YAML.

## CLI help: essential vs advanced

### Global range constraints

`global_rmsd_range`, `global_e2e_range`, `global_rg_range`, and
`global_token_ranges` are rejected for GRPO runs. The stand-alone policy sampler
can restrict proposals, but its rejection/clamping law is not represented by
`log_prob`; using its unrestricted likelihood would give incorrect policy
gradients. Configuration validation and the training entry points enforce this
restriction before launching MD.

Both current and behavior policy networks use evaluation mode during GRPO
likelihood scoring, so dropout does not change the importance ratios. Autograd
remains enabled for the current policy.

`gespo train --help` shows a curated **essential** flag set; `gespo train --help-all`
shows every flag (~240). All flags are always accepted — `--help` only hides the
advanced ones. Shortcuts: `--mode {generative,path-sampling,standard}` and
`--solvent {explicit,implicit,gbsa,gbn2}` expand to the underlying booleans;
explicitly-passed flags win over a shortcut.

## Configuration: precedence and YAML sections

Precedence (highest first): **explicit CLI flag → YAML value → dataclass default**.
A flag you did not pass does *not* override YAML (the None-sentinel fix). Save the
fully-resolved config from any run at `<snapshot_dir>/effective_config.yaml`.

YAML sections (`gespo config template` emits a starting file):

| Section | Keys become | Notes |
|---|---|---|
| `endpoints` | `pdb_a`→`pdb`, `pdb_b` | |
| `training` | `mode`, `episodes`, `grp_size`, `platform`, `gen_device` | `mode` sets the loop flags |
| `md` | `n_md_steps`, `temperature`, `solvent`, forcefields, … | `solvent` sets implicit/explicit |
| `policy` | `torsion_ar_mode`, `torsion_ar_checkpoint`, `lr` | |
| `geom2vec` | prefixed with `geom2vec_` | `checkpoint`→`geom2vec_checkpoint` |
| `rewards` | **full field names** (no prefix) | `loo_vamp_reward_weight`, … |
| `steering` | prefixed with `steer_` | `k_rmsd`→`steer_k_rmsd` |
| `committor` | prefixed with `committor_` | path-sampling only |
| `io` | `snapshot_dir`, `checkpoint_dir`, `checkpoint_interval`, `resume` | |

Any dataclass field may also be set as a **flat top-level key** (the long tail —
`ligand_smiles`, `retrieval_candidates`, `projection`, …); see
[`../examples/protein_ligand_trypsin.yaml`](../examples/protein_ligand_trypsin.yaml).
Unknown keys/sections warn (or raise under `gespo config validate --strict`).
