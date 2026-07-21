# End-to-end workflow

A worked walkthrough of the full pipeline, matching
[`../examples/protein_ligand_trypsin/`](../examples/protein_ligand_trypsin/).
Set `PYTHONPATH` to the repo and activate an OpenMM environment first.

## Stage 1 — seed

```bash
python -m gespo seed metad \
  --pdb setup/trypsin_ben_explicit_ready.pdb \
  --ligand-smiles "c1ccc(cc1)C(=N)N" \
  --cv-set holo --explicit-solvent \
  --n-walkers 4 --device cuda:0,cuda:1,cuda:2,cuda:3 \
  --production-ns 25 --out-dir metad_seeds
# -> metad_seeds/walker_{1..4}/production.dcd
```

See [`seeding.md`](seeding.md) for the CV sets and knobs.

## Stage 2 — pretrain the policy

```bash
python -m gespo pretrain \
  --source metad_seeds/walker_1,metad_seeds/walker_2,metad_seeds/walker_3,metad_seeds/walker_4 \
  --pdb setup/trypsin_ben_explicit_ready.pdb \
  --out torsion_ar_holo_pretrain.pt \
  --global-token-set rmsd_e2e_pocket_full \
  --ligand-smiles "c1ccc(cc1)C(=N)N" \
  --epochs 20 --batch-size 64 --lr 1e-4 --device cuda:0
# -> torsion_ar_holo_pretrain.pt
```

`--global-token-set` chooses the global CVs the policy conditions on
([`reference.md`](reference.md#global-token-sets)). With `--ligand-smiles`, the
policy also models the ligand's rotatable dihedrals. `--torsion-mode
backbone+sidechain` adds sidechain χ angles.

## Stage 3 — train

```bash
python -m gespo train \
  --pdb setup/trypsin_ben_explicit_ready.pdb \
  --ligand-smiles "c1ccc(cc1)C(=N)N" \
  --generative-loop --explicit-solvent \
  --torsion-ar-checkpoint torsion_ar_holo_pretrain.pt \
  --projection geom2vec_vamp --geom2vec-ckpt /path/to/backbone.pth \
  --replay-only-dcd-dir metad_seeds/walker_1,metad_seeds/walker_2,metad_seeds/walker_3,metad_seeds/walker_4 \
  --episodes 15 --grp-size 12 --n-md-steps 1000000 \
  --snapshot-dir runs/run_01/run_snaps --checkpoint-dir runs/run_01/run_snaps/ckpts
  # ... plus reward / steering / geom2vec knobs — see the example script
```

The full flag set is in
[`../examples/protein_ligand_trypsin/03_train_r1.sh`](../examples/protein_ligand_trypsin/03_train_r1.sh),
and the same run as a single `--config` file is in
[`../examples/protein_ligand_trypsin.yaml`](../examples/protein_ligand_trypsin.yaml):

```bash
python -m gespo train --config examples/protein_ligand_trypsin.yaml \
  --pdb setup/trypsin_ben_explicit_ready.pdb \
  --geom2vec-ckpt /path/to/backbone.pth \
  --replay-only-dcd-dir metad_seeds/walker_1,metad_seeds/walker_2,metad_seeds/walker_3,metad_seeds/walker_4
```

### Outputs (per round)

```
runs/run_01/run_snaps/
  effective_config.yaml         # every resolved parameter (reproducibility)
  simulation_topology.pdb       # the built system (carry forward to round 2)
  endpoint_A.pdb
  epoch_0001/ .. epoch_00NN/     # per-epoch, per-walker DCDs + feature/reward records
  ckpts/ckpt_ep_XXXX.pt          # RL checkpoints (for --resume)
  geom2vec_vamp/                 # embedding cache + fitted VAMP lobe
```

Watch progress via the `[Epoch e/E][Phase A..F]` banners (see
[`overview.md`](overview.md#the-rl-loop-one-epoch-at-a-time)).

## Iterating: the multi-round protocol

Each round expands the ensemble; the next round learns from it. Round *n+1*:

1. **Re-pretrain** the policy *cumulatively* — `--source` = the original seeds **+**
   the previous round's epoch DCDs (`runs/run_0n/run_snaps/epoch_*`), a few epochs,
   warm-started.
2. **Grow the replay pool** — `--replay-only-dcd-dir` gains the previous round's
   epoch DCDs.
3. **Carry the topology forward** — copy `simulation_topology.pdb` into the new
   snapshot dir so the solvated system is identical.
4. **Train** as before, into a new `runs/run_0(n+1)/`.

[`../examples/protein_ligand_trypsin/04_train_r2.sh`](../examples/protein_ligand_trypsin/04_train_r2.sh)
implements exactly this.

## Programmatic equivalent

```python
import gespo

gespo.generate_seeds(method="metad", pdb="holo.pdb", out_dir="metad_seeds",
                     cv_set="holo", ligand_smiles="c1ccc(cc1)C(=N)N", n_walkers=4)
gespo.pretrain_torsion_ar(source_path="metad_seeds/", out_path="ar.pt",
                          pdb_path="holo.pdb", global_token_set="rmsd_e2e_pocket_full",
                          ligand_smiles="c1ccc(cc1)C(=N)N")
gespo.train("examples/protein_ligand_trypsin.yaml")   # or a dict / TrainingConfig
```
