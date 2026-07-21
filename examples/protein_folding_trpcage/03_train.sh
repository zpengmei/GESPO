#!/usr/bin/env bash
# 03_train.sh — GESPO generative loop for Trp-cage (implicit solvent).
#
# The modern single-command reproduction: bootstrap from the pretrained TorsionAR
# policy + a geom2vec-VAMP backbone (no diffusion model). geom2vec-VAMP discovers
# metastable states online; LOO-VAMP + VAMP-bin rewards drive the policy to expand
# the sampled conformational ensemble; retrieval-steered replay resets seed rollouts
# from under-sampled states.
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────────────────
GESPO_REPO="${GESPO_REPO:-/path/to/GESPO}"
PDB="${PDB:-trpcage.pdb}"
GEOM2VEC_CKPT="${GEOM2VEC_CKPT:-/path/to/geom2vec/tensornet_l3_h64_rbf32_r5.pth}"
AR_CKPT="${AR_CKPT:-torsion_ar_round0.pt}"
DEVICE="${DEVICE:-cuda:0}"
# ─────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${GESPO_REPO}:${PYTHONPATH:-}"

WORK_DIR="runs/round1_implicit"
SNAPSHOT_DIR="${WORK_DIR}/run_snaps"
CKPT_DIR="${SNAPSHOT_DIR}/ckpts"
mkdir -p "${CKPT_DIR}"

exec python -m gespo train \
  --pdb "${PDB}" \
  --generative-loop \
  --implicit-solvent \
  --platform CUDA \
  --gen-device "${DEVICE}" \
  --episodes 27 \
  --grp-size 8 \
  --n-md-steps 75000 \
  --record-stride 250 \
  --temperature 300 \
  --minimize-iters 500 \
  --lr 1e-4 \
  --ent-coef 0.05 \
  --kl-coef 0.5 \
  --grpo-iters 10 \
  --sync-interval 3 \
  --energy-threshold 1e12 \
  --max-resample-attempts 6 \
  --no-anchoring-round \
  --torsion-ar-checkpoint "${AR_CKPT}" \
  --projection geom2vec_vamp \
  --geom2vec-ckpt "${GEOM2VEC_CKPT}" \
  --geom2vec-cache-root "${SNAPSHOT_DIR}" \
  --geom2vec-embed-stride 2 \
  --geom2vec-lagtime 150 \
  --geom2vec-dim 16 \
  --geom2vec-vamp-epochs 10 \
  --geom2vec-vamp-batch-size 1024 \
  --geom2vec-vamp-learning-rate 5e-4 \
  --geom2vec-vamp-epsilon 1e-6 \
  --geom2vec-vamp-inverse-mode regularize \
  --retrieval-candidates 4096 \
  --retrieval-topk 32 \
  --retrieval-max-resamples 3 \
  --retrieval-cache-max-frames 100000 \
  --steer-replay-starts \
  --steer-global-steps 10000 \
  --steer-global-updates 25 \
  --steer-torsion-steps 2000 \
  --steer-torsion-updates 25 \
  --steer-release-steps 0 \
  --steer-refine-minimize-iters 0 \
  --steer-global-lambda-during-torsion 1 \
  --steer-k-rmsd 10000 \
  --steer-k-e2e 5000 \
  --steer-k-torsion 200 \
  --steer-replay-verbose \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --checkpoint-interval 3 \
  --plot-freq 1
