# Validation environment

`validation-macos-arm64-py313.txt` records exact versions used for the local
publication checks on Apple Silicon with CPython 3.13.13. It includes the
runtime, OpenMM, test, and build dependency closure. It is a platform-specific
validation snapshot, not a validated CUDA production environment.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r environments/validation-macos-arm64-py313.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest gespo/tests -q
```

Geom2Vec backbones and protein–ligand parameterization dependencies are separate
inputs and must be exported from the environment of each production campaign.
Record the Geom2Vec commit and checkpoint SHA-256 as well as this repository's SHA.
