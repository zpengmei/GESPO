#!/usr/bin/env bash
# 03_train_r1.sh — GESPO round 1, holo trypsin + benzamidine.
#
# Binding-pathway exploration with:
#   - geom2vec-VAMP projection + LOO-VAMP / VAMP-bin / MSM-info-gain rewards
#   - retrieval-based replay resets steered by targeted MD
#   - a λ-toggled ligand↔pocket flat-bottom restraint: ON during steering (so the
#     policy can't reward-hack by dissociating BEN), OFF during production rollouts
#     (so production dynamics stay unbiased)
# The replay pool is the metaD seeds from 01_seed.sh.
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────────────────
GESPO_REPO="${GESPO_REPO:-/path/to/GESPO}"
PDB="${PDB:-setup/trypsin_ben_explicit_ready.pdb}"
GEOM2VEC_CKPT="${GEOM2VEC_CKPT:-/path/to/geom2vec/tensornet_l3_h64_rbf32_r5.pth}"
LIGAND_SMILES="${LIGAND_SMILES:-c1ccc(cc1)C(=N)N}"
AR_CKPT="${AR_CKPT:-torsion_ar_holo_pretrain.pt}"
DEVICE="${DEVICE:-cuda:0}"
# ─────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${GESPO_REPO}:${PYTHONPATH:-}"

WORK_DIR="runs/run_01"
SNAPSHOT_DIR="${WORK_DIR}/run_snaps"
CKPT_DIR="${SNAPSHOT_DIR}/ckpts"
mkdir -p "${CKPT_DIR}"

REPLAY_DIRS="metad_seeds/walker_1,metad_seeds/walker_2,metad_seeds/walker_3,metad_seeds/walker_4"

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
  --replay-only-dcd-dir "${REPLAY_DIRS}" \
  --ligand-pocket-restraint \
  --ligand-pocket-restraint-radius-nm 0.6 \
  --ligand-pocket-restraint-k 1000 \
  --torsion-ar-checkpoint "${AR_CKPT}" \
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
