#!/usr/bin/env bash
# 01_seed.sh — generate seed trajectories for holo trypsin + benzamidine.
#
# Well-tempered metadynamics with the "holo" CV set:
#   CV1 = pocket Cα RMSD          (loop opening)
#   CV2 = ligand↔pocket COM dist  (deep ↔ surface poses; ligand stays bound via
#                                  a flat-bottom restraint, r0=1.0 nm, k=1000)
# 4 walkers share a bias directory (parallel metaD). Output feeds 02_pretrain.sh
# (AR pretraining) and 03_train_r1.sh (replay pool).
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────────────────
GESPO_REPO="${GESPO_REPO:-/path/to/GESPO}"                  # this repository
PDB="${PDB:-setup/trypsin_ben_explicit_ready.pdb}"          # solvated holo complex (with BEN)
LIGAND_SMILES="${LIGAND_SMILES:-c1ccc(cc1)C(=N)N}"          # benzamidine
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"           # one GPU per walker
N_WALKERS="${N_WALKERS:-4}"
PRODUCTION_NS="${PRODUCTION_NS:-25}"
# Activate your OpenMM environment first, e.g.:  conda activate openmm
# ─────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${GESPO_REPO}:${PYTHONPATH:-}"

exec python -m gespo seed metad \
  --pdb "${PDB}" \
  --ligand-smiles "${LIGAND_SMILES}" \
  --cv-set holo \
  --explicit-solvent \
  --n-walkers "${N_WALKERS}" \
  --device "${DEVICES}" \
  --production-ns "${PRODUCTION_NS}" \
  --lig-restraint-r0-nm 1.0 \
  --lig-restraint-k 1000 \
  --out-dir metad_seeds
# → metad_seeds/walker_{1..N}/production.dcd
