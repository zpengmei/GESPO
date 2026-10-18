# GESPO examples

Two end-to-end, **real** recipes plus a minimal YAML. Each recipe follows the
GESPO pipeline:

```
 seed  ──▶  pretrain  ──▶  train  ──▶  (iterate rounds)
 gespo seed    gespo pretrain   gespo train
```

| Example | System | Solvent | Highlights |
|---|---|---|---|
| [`protein_folding_trpcage/`](protein_folding_trpcage/) | Trp-cage (20-residue miniprotein) | implicit (GBn2) | folding CVs, introductory single-round workflow |
| [`protein_ligand_trypsin/`](protein_ligand_trypsin/) | trypsin + benzamidine (holo) | explicit (TIP3P) | ligand SMILES→GAFF, pocket restraint, 2 rounds |
| [`generative_min.yaml`](generative_min.yaml) | — | — | smallest possible `--config` file |
| [`protein_ligand_trypsin.yaml`](protein_ligand_trypsin.yaml) | trypsin + benzamidine | explicit | the holo recipe expressed as one `--config` file |

## What you must provide (the input contract)

The recipes are **not self-contained** — molecular systems are large research
artifacts, so nothing is bundled. Before running, supply:

1. **A prepared PDB** — your solvated (explicit) or hydrogen-added (implicit)
   structure. For a protein-ligand complex, the ligand should be present as a
   residue (e.g. `BEN`/`UNK`). See each recipe's README for what it expects.
2. **A geom2vec backbone checkpoint** (`*.pth`) — the pretrained representation
   model used by the `geom2vec_vamp` projection. Point `GEOM2VEC_CKPT` at it.
3. **Nothing else** — the seed trajectories are produced by step `01_seed.sh`
   (that is the whole point of `gespo seed`).

## How to run

Each recipe is a numbered sequence of shell scripts. Edit the `EDIT THESE`
block at the top of each (paths, device, conda env), then run in order:

```bash
cd protein_ligand_trypsin
bash 01_seed.sh        # gespo seed  -> metad_seeds/walker_*/production.dcd
bash 02_pretrain.sh    # gespo pretrain -> torsion_ar_*.pt
bash 03_train_r1.sh    # gespo train (round 1)
bash 04_train_r2.sh    # gespo train (round 2: retrained policy + grown replay pool)
```

The scripts assume an OpenMM-enabled Python environment on a CUDA GPU. They set
`PYTHONPATH` to this repo and expect `python -m gespo` to resolve. See
[`../docs/`](../docs/) for concepts, inputs, the token-set catalog, and the
reward-term reference.
