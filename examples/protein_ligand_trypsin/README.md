# Protein–ligand: trypsin + benzamidine (holo)

Explore the **binding pathway** of benzamidine (BEN) in the trypsin S1 pocket
with GESPO. This is the protein–ligand reference recipe: it exercises SMILES→GAFF
ligand parameterization, ligand torsion tokens, pocket auto-detection, and the
λ-toggled ligand–pocket restraint.

## Inputs you must provide

| Variable | What | Example |
|---|---|---|
| `GESPO_REPO` | Path to this repository | `/path/to/GESPO` |
| `PDB` | Solvated holo complex (protein + BEN + water box). BEN present as a residue. | `setup/trypsin_ben_explicit_ready.pdb` |
| `GEOM2VEC_CKPT` | Pretrained geom2vec backbone (`.pth`) | `tensornet_l3_h64_rbf32_r5.pth` |
| `LIGAND_SMILES` | Ligand SMILES (benzamidine) | `c1ccc(cc1)C(=N)N` |

To prepare the complex: dock/soak the ligand, add hydrogens, solvate in a TIP3P
box with neutralizing ions, and briefly equilibrate — any standard OpenMM/Amber
setup works, as long as the ligand residue is present in the final PDB.

## Run

```bash
export GESPO_REPO=/path/to/GESPO
export GEOM2VEC_CKPT=/path/to/tensornet_l3_h64_rbf32_r5.pth
conda activate openmm            # your OpenMM environment

bash 01_seed.sh        # 4 metaD walkers -> metad_seeds/walker_*/production.dcd
bash 02_pretrain.sh    # TorsionAR (ligand torsion tokens) -> torsion_ar_holo_pretrain.pt
bash 03_train_r1.sh    # GESPO round 1 -> runs/run_01/
bash 04_train_r2.sh    # GESPO round 2 (retrained policy + grown replay pool)
```

`01_seed.sh` fans 4 walkers across `cuda:0..3`; reduce `N_WALKERS`/`DEVICES` if
you have fewer GPUs. Each round writes per-epoch, per-walker DCDs under
`runs/run_0N/run_snaps/epoch_*/`, RL checkpoints under `.../ckpts/`, and
per-epoch analysis plots.

## What each step does

- **Seed** — well-tempered metadynamics on the `holo` CV set (pocket Cα RMSD +
  ligand↔pocket COM distance). A flat-bottom restraint keeps BEN in the pocket so
  the seeds sample *bound* poses rather than dissociation.
- **Pretrain** — a TorsionAR policy over backbone torsions **plus** benzamidine's
  rotatable dihedrals, conditioned on the `rmsd_e2e_pocket_full` global tokens.
- **Train** — the GESPO RL loop: geom2vec-VAMP state discovery, VAMP-bin +
  MSM-info-gain diversity rewards, retrieval-steered replay resets. The ligand
  restraint is ON only while steering.
- **Round 2** — re-pretrain the policy on seeds + round-1 trajectories, grow the
  replay pool, and continue — the iterative enhanced-sampling protocol.

For the apo (no-ligand) variant, drop `--ligand-smiles`/`--ligand-pocket-restraint`,
use `--cv-set apo --pocket-ref-pdb <holo.pdb>` for seeding, and the
`rmsd_e2e_pocket3` token set for pretraining. See [`../../docs/ligands.md`](../../docs/ligands.md).
