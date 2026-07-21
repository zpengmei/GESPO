#!/usr/bin/env bash
# 02_pretrain.sh — pretrain the TorsionAR policy on the Trp-cage folding seeds.
#
# Token set rmsd_e2e_rg = [global RMSD, end-to-end distance, radius of gyration]
# (no ligand → no ligand torsion tokens).
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────────────────
GESPO_REPO="${GESPO_REPO:-/path/to/GESPO}"
PDB="${PDB:-trpcage.pdb}"
DEVICE="${DEVICE:-cuda:0}"
# ─────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${GESPO_REPO}:${PYTHONPATH:-}"

SEEDS="metad_seeds/walker_1,metad_seeds/walker_2,metad_seeds/walker_3,metad_seeds/walker_4"

exec python -m gespo pretrain \
  --source "${SEEDS}" \
  --pdb "${PDB}" \
  --out torsion_ar_round0.pt \
  --global-token-set rmsd_e2e_rg \
  --stride 1 \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-4 \
  --device "${DEVICE}"
# → torsion_ar_round0.pt  (used by 03_train.sh)
