#!/usr/bin/env bash
# 01_seed.sh — generate folding seeds for Trp-cage (implicit solvent).
#
# Well-tempered metadynamics with the "folding" CV set:
#   CV1 = global Cα RMSD to the folded reference
#   CV2 = end-to-end (N-term ↔ C-term) distance
# This drives fold/unfold transitions so the seeds span the folding coordinate.
# Trp-cage is small, so implicit (GBn2) solvent is used — no box, no barostat.
set -euo pipefail

# ── EDIT THESE ───────────────────────────────────────────────────────────
GESPO_REPO="${GESPO_REPO:-/path/to/GESPO}"
PDB="${PDB:-trpcage.pdb}"                     # folded Trp-cage, hydrogens added
DEVICES="${DEVICES:-cuda:0,cuda:1}"           # implicit runs are cheap; a couple GPUs suffice
N_WALKERS="${N_WALKERS:-4}"
PRODUCTION_NS="${PRODUCTION_NS:-25}"
# Activate your OpenMM environment first, e.g.:  conda activate openmm
# ─────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${GESPO_REPO}:${PYTHONPATH:-}"

exec python -m gespo seed metad \
  --pdb "${PDB}" \
  --cv-set folding \
  --implicit-solvent \
  --n-walkers "${N_WALKERS}" \
  --device "${DEVICES}" \
  --production-ns "${PRODUCTION_NS}" \
  --cv1-max-nm 1.0 \
  --cv2-max-nm 4.0 \
  --out-dir metad_seeds
# → metad_seeds/walker_{1..N}/production.dcd
#
# Alternative: high-temperature unfolding instead of metaD:
#   python -m gespo seed high-temp --pdb "${PDB}" --temperature 400 \
#     --implicit-solvent --n-walkers 4 --production-ns 50 --out-dir highT_seeds
