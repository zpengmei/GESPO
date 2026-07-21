#!/usr/bin/env bash
# 02_pretrain.sh — pretrain the TorsionAR policy on the metaD seeds.
#
# Token set rmsd_e2e_pocket_full = 5 global tokens
#   (rmsd_nm, e2e_nm, pocket_rmsd_nm, n_contacts_scaled, lig_pocket_dist_nm)
# plus RDKit-detected ligand torsion tokens (benzamidine has 2 rotatable bonds).
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────────────────
GESPO_REPO="${GESPO_REPO:-/path/to/GESPO}"
PDB="${PDB:-setup/trypsin_ben_explicit_ready.pdb}"
LIGAND_SMILES="${LIGAND_SMILES:-c1ccc(cc1)C(=N)N}"
DEVICE="${DEVICE:-cuda:0}"
# ─────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${GESPO_REPO}:${PYTHONPATH:-}"

SEEDS="metad_seeds/walker_1,metad_seeds/walker_2,metad_seeds/walker_3,metad_seeds/walker_4"

exec python -m gespo pretrain \
  --source "${SEEDS}" \
  --pdb "${PDB}" \
  --out torsion_ar_holo_pretrain.pt \
  --global-token-set rmsd_e2e_pocket_full \
  --ligand-smiles "${LIGAND_SMILES}" \
  --stride 1 \
  --epochs 20 \
  --batch-size 64 \
  --lr 1e-4 \
  --device "${DEVICE}"
# → torsion_ar_holo_pretrain.pt  (used by 03_train_r1.sh)
