# Seeding: `gespo seed`

The initial-data stage. `gespo seed` runs short biased/high-temperature MD to
produce trajectories that (1) pretrain the TorsionAR policy and (2) seed the replay
pool for `gespo train`. Three methods share one system-building core and support
both explicit and implicit solvent.

```
gespo seed metad      --pdb ... --out-dir seeds/ ...   # well-tempered metadynamics
gespo seed high-temp  --pdb ... --out-dir seeds/ ...   # high-temperature MD
gespo seed rmsd-bias  --pdb ... --out-dir seeds/ ...   # RMSD-biased MD
```

All methods write the standard layout `<out-dir>/walker_i/production.dcd`, which
feeds `gespo pretrain --source` and `gespo train --replay-only-dcd-dir` directly.

Programmatic API (walker methods): `gespo.generate_seeds(method="metad", pdb=...,
out_dir=..., cv_set="holo", ligand_smiles=..., n_walkers=4)`.

## `metad` — well-tempered metadynamics

Biases two collective variables to flatten the free-energy surface and sweep the
CVs. Choose the CV pair with `--cv-set`:

| `--cv-set` | CV1 | CV2 | Use for |
|---|---|---|---|
| `holo` | pocket Cα RMSD | ligand↔pocket COM distance | protein–ligand (bound poses) |
| `apo` | pocket Cα RMSD | global Cα RMSD | apo pocket + global motion |
| `folding` | global Cα RMSD | end-to-end distance | fold/unfold |

- **holo** needs the ligand in `--pdb` (and `--ligand-smiles` to parameterize it);
  a flat-bottom restraint keeps the ligand bound so seeds sample bound poses rather
  than dissociation (`--lig-restraint-r0-nm`, `--lig-restraint-k`; disable with
  `--no-lig-restraint`).
- **apo** detects the pocket from a holo reference: pass `--pocket-ref-pdb <holo.pdb>`.
- CV bounds/width default to the proven values; override with `--cv1-max-nm`,
  `--cv2-min-nm`, `--cv2-max-nm`. metaD knobs: `--bias-factor` (γ), `--hill-height`,
  `--hill-freq`.

Multiple walkers share one bias directory (`--bias-dir`, default `<out-dir>/bias`)
so their Gaussians accumulate collectively (parallel metaD). Fan out with
`--n-walkers N --device cuda:0,cuda:1,...` (one GPU per walker).

## `high-temp` — high-temperature MD

Heats the system from 298 K to `--temperature` (staged, 10 × 200 ps) and runs
production there — a simple way to generate diverse conformations. Supports
`--restart` (resume from checkpoint) and, for protein–ligand, an optional
single-residue ligand restraint (`--ligand-pocket-restraint --pocket-seq-pos`).

## `rmsd-bias` — RMSD-biased MD

Ramps a harmonic RMSD restraint from 0 → `--r0-nm-max` to pull a protein open,
optionally emitting a stratified multi-model `start_pool.pdb`. This wraps the
battle-tested implementation in `gespo/bootstrap_cli.py`; `gespo seed rmsd-bias` and
`python -m gespo.bootstrap_cli rmsd-bias` are the same code. Single trajectory (no
walker fan-out).

## Explicit vs implicit solvent

- `--explicit-solvent` (default): re-solvate a fresh TIP3P box + PME + barostat.
- `--implicit-solvent`: GBn2 implicit model, no box/barostat — right for small
  proteins (the Trp-cage example seeds this way).

Forcefields are configurable: `--forcefield-protein`, `--forcefield-water`,
`--forcefield-implicit`.

## Examples

- Protein–ligand (holo trypsin): [`../examples/protein_ligand_trypsin/01_seed.sh`](../examples/protein_ligand_trypsin/01_seed.sh)
- Folding (Trp-cage, implicit): [`../examples/protein_folding_trpcage/01_seed.sh`](../examples/protein_folding_trpcage/01_seed.sh)
