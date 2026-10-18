#!/usr/bin/env bash
# 04_train_r2.sh — GESPO round 2 (the multi-round protocol).
#
# Round 2 differs from round 1 in three ways:
#   1. The AR policy is *re-pretrained cumulatively* on the seeds + round-1 epoch
#      DCDs (fewer epochs, warm start) → torsion_ar_holo_r1_retrain.pt
#   2. The replay pool grows to include round-1 epoch trajectories.
#   3. simulation_topology.pdb is carried forward so the solvated system is
#      identical across rounds.
# Everything else matches round 1 (including the λ-toggled ligand restraint).
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────────────────
GESPO_REPO="${GESPO_REPO:-/path/to/GESPO}"
PDB="${PDB:-setup/trypsin_ben_explicit_ready.pdb}"
GEOM2VEC_CKPT="${GEOM2VEC_CKPT:-/path/to/geom2vec/tensornet_l3_h64_rbf32_r5.pth}"
LIGAND_SMILES="${LIGAND_SMILES:-c1ccc(cc1)C(=N)N}"
DEVICE="${DEVICE:-cuda:0}"
# ─────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${GESPO_REPO}:${PYTHONPATH:-}"

R1_SNAPS="runs/run_01/run_snaps"
WORK_DIR="runs/run_02"
SNAPSHOT_DIR="${WORK_DIR}/run_snaps"
CKPT_DIR="${SNAPSHOT_DIR}/ckpts"
mkdir -p "${CKPT_DIR}"

# Biased seeds are retrieval-only; unbiased production also trains the kinetic model.
SEED_DIRS="metad_seeds/walker_1,metad_seeds/walker_2,metad_seeds/walker_3,metad_seeds/walker_4"
HISTORY_DIRS=""
for ep_dir in "${R1_SNAPS}"/epoch_*; do
  if [ -d "${ep_dir}" ]; then
    HISTORY_DIRS="${HISTORY_DIRS:+${HISTORY_DIRS},}${ep_dir}"
  fi
done
if [ -z "${HISTORY_DIRS}" ]; then
  echo "No round-1 epoch directories found under ${R1_SNAPS}" >&2
  exit 1
fi
SOURCE_DIRS="${SEED_DIRS},${HISTORY_DIRS}"

# Carry the solvated topology forward so the system matches round 1 exactly.
cp "${R1_SNAPS}/simulation_topology.pdb" "${SNAPSHOT_DIR}/simulation_topology.pdb"
PDB="${R1_SNAPS}/simulation_topology.pdb"

# ── Step A: re-pretrain the AR policy cumulatively (seeds + round-1 epochs) ──
AR_R2="torsion_ar_holo_r1_retrain.pt"
python -m gespo pretrain \
  --source "${SOURCE_DIRS}" \
  --pdb "${PDB}" \
  --out "${AR_R2}" \
  --global-token-set rmsd_e2e_pocket_full \
  --ligand-smiles "${LIGAND_SMILES}" \
  --stride 1 \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-4 \
  --device "${DEVICE}"

# ── Step B: round-2 training with the retrained policy + grown replay pool ──
exec python -m gespo train \
  --pdb "${PDB}" \
  --ligand-smiles "${LIGAND_SMILES}" \
  --generative-loop \
  --explicit-solvent \
  --platform CUDA \
  --gen-device "${DEVICE}" \
  --enable-multi-gpu \
  --episodes 15 \
  --grp-size 12 \
  --n-md-steps 1000000 \
  --record-stride 4000 \
  --temperature 298 \
  --forcefield-protein amber14-all.xml \
  --forcefield-water amber14/tip3p.xml \
  --minimize-iters 500 \
  --equilibration-steps 5000 \
  --lr 5e-5 \
  --ent-coef 0.15 \
  --kl-coef 0.1 \
  --vamp-bin-entropy-coef -1 \
  --msm-ig-reward-coef 0.5 \
  --start-distance-reward-coef 0.0 \
  --pcca-n-macrostates 7 \
  --grpo-iters 20 \
  --sync-interval 1 \
  --energy-threshold 1e12 \
  --max-resample-attempts 6 \
  --no-anchoring-round \
  --replay-only-dcd-dir "${SEED_DIRS}" \
  --history-dcd-dir "${HISTORY_DIRS}" \
  --ligand-pocket-restraint \
  --ligand-pocket-restraint-radius-nm 0.6 \
  --ligand-pocket-restraint-k 1000 \
  --torsion-ar-checkpoint "${AR_R2}" \
  --projection geom2vec_vamp \
  --geom2vec-ckpt "${GEOM2VEC_CKPT}" \
  --geom2vec-cache-root "${SNAPSHOT_DIR}" \
  --geom2vec-embed-stride 2 \
  --geom2vec-lagtime 50 \
  --geom2vec-dim 16 \
  --geom2vec-vamp-epochs 10 \
  --geom2vec-vamp-batch-size 1024 \
  --geom2vec-vamp-learning-rate 5e-4 \
  --geom2vec-vamp-epsilon 1e-6 \
  --geom2vec-vamp-inverse-mode regularize \
  --retrieval-candidates 4096 \
  --retrieval-topk 32 \
  --retrieval-max-resamples 3 \
  --retrieval-cache-max-frames 200000 \
  --steer-replay-starts \
  --steer-global-steps 5000 \
  --steer-global-updates 25 \
  --steer-torsion-steps 2000 \
  --steer-torsion-updates 25 \
  --steer-release-steps 1000 \
  --steer-refine-minimize-iters 500 \
  --steer-global-lambda-during-torsion 0.2 \
  --steer-k-rmsd 2000 \
  --steer-k-e2e 1000 \
  --steer-k-torsion 100.0 \
  --steer-replay-verbose \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --checkpoint-interval 3 \
  --plot-freq 1
