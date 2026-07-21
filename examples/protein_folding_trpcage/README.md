# Protein folding: Trp-cage

Expand the conformational ensemble of the **Trp-cage** miniprotein (20 residues,
TC5b) with GESPO, in **implicit solvent**. This is the smallest, fastest recipe —
a good first run to confirm your environment works end to end.

## Inputs you must provide

| Variable | What | Example |
|---|---|---|
| `GESPO_REPO` | Path to this repository | `/path/to/GESPO` |
| `PDB` | Folded Trp-cage with hydrogens (single conformer) | `trpcage.pdb` |
| `GEOM2VEC_CKPT` | Pretrained geom2vec backbone (`.pth`) | `tensornet_l3_h64_rbf32_r5.pth` |

Any folded Trp-cage structure with hydrogens works. If yours lacks hydrogens, add
them first (e.g. with PDBFixer). No solvent box is needed — the recipe uses the
GBn2 implicit-solvent model.

## Run

```bash
export GESPO_REPO=/path/to/GESPO
export GEOM2VEC_CKPT=/path/to/tensornet_l3_h64_rbf32_r5.pth
conda activate openmm

bash 01_seed.sh      # folding metaD -> metad_seeds/walker_*/production.dcd
bash 02_pretrain.sh  # TorsionAR -> torsion_ar_round0.pt
bash 03_train.sh     # GESPO generative loop -> runs/round1_implicit/
```

## What each step does

- **Seed** — well-tempered metadynamics on the `folding` CV set (global Cα RMSD +
  end-to-end distance), sampling fold/unfold transitions to give the policy a
  diverse starting ensemble.
- **Pretrain** — a backbone TorsionAR policy conditioned on `rmsd_e2e_rg` global
  tokens (RMSD, end-to-end distance, radius of gyration).
- **Train** — the GESPO generative loop with a geom2vec-VAMP projection. Over 27
  episodes it discovers metastable states online and rewards the policy for
  broadening the sampled ensemble. Outputs land in `runs/round1_implicit/run_snaps/`
  (per-epoch DCDs, checkpoints, Ramachandran/analysis plots).

To iterate, re-pretrain on the accumulated trajectories and run another round —
see the multi-round pattern in [`../protein_ligand_trypsin/04_train_r2.sh`](../protein_ligand_trypsin/04_train_r2.sh)
and [`../../docs/workflow.md`](../../docs/workflow.md).
