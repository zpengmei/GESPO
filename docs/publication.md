# Publication and reproducibility status

The development implementation is `0.2.0.dev0`. The review baseline is commit
`d73b05ae2481786fe556360ed78805c5df9fa633`. Earlier PR descriptions associate
the historical campaigns with `d3b1af01bbfc094b6ebf69a8f565d6d1a3cf0beb`, but that
association has not been verified against result manifests. Neither commit is
being presented as a certified reproduction snapshot of every defense figure.

## Corrected implementation

- GRPO disables dropout in both policy networks while preserving gradients.
- Range-constrained global proposals are rejected during GRPO until their
  normalized likelihood and boundary masses are implemented.
- Every seeded DCD has its actual `simulation_topology.pdb`; restarts reuse it.
- The round-two recipe imports unbiased production as kinetic history and
  quarantines biased seeds as retrieval-only data.
- Built-in diagnostics and legacy boundary plugins are included in wheels.
- CI checks the unit suite and installed-wheel imports on Python 3.11 and 3.12.
  A workflow file is not evidence that remote CI has passed; consult the run.

The optimizer differs from the historical implementation. New campaigns require
fresh validation and coefficient selection; historical coverage numbers do not
establish the performance of this development version.

Local validation passed 125 tests, built and imported the wheel outside the
checkout, and ran a 2 ps implicit-solvent CPU seed-to-pretraining smoke test
(four saved frames, one MLE epoch). The six unit-test warnings come from
single-frame HELANAL fixtures. This is workflow validation, not a production
campaign or a test of GPU parallelism, ligand parameterization, or Geom2Vec fitting.

To repeat the small physical workflow check in a fresh output directory:

```bash
OPENMM_CPU_THREADS=2 python scripts/smoke_seed_pretrain.py --out-dir /tmp/gespo-smoke-new
```

The smoke test skips equilibration deliberately. Normal metadynamics seeding
retains the default `--equilibration-ns 2`; zero should not be used to claim
an equilibrated production system.

## Regenerating coverage summaries

Export one CSV per system with columns
`method,replicate,budget_ns,cv_x,cv_y`. Each row represents an observed production
frame. `budget_ns` must use the same accounting rule for every method and record
the aggregate unbiased production budget when the frame was acquired. Keep
seeding, realization, training time, and total wall time as separate costs.
Do not infer budget from concatenated frame order across parallel walkers.

Supply the original fixed grid bounds explicitly; the tool does not select
bounds from either method's data:

```bash
python scripts/coverage_benchmark.py --csv frames.csv \
  --x-range X_MIN X_MAX --y-range Y_MIN Y_MAX --grid-size 60 --out-dir coverage
```

Replace the range placeholders with the actual values in the stated CV units.
The script writes `coverage.csv`, a coverage-curve PNG, and `provenance.json`
with the input SHA-256, grid definition, and excluded-frame counts. It does not
regenerate the historical figure without the original inputs and grid definition.

Before presenting a comparative performance claim, archive the prepared systems,
force-field and ligand charge choices, seed corpora, model checkpoints, trajectory
manifests, all methods' code SHAs and effective configurations, grid definitions,
analysis scripts, and environment versions. Run independent campaign replicates
and controls that separate policy optimization from pretraining/retrieval and
the individual reward terms, or explicitly limit the claim to the observed
campaigns. The public materials currently do not establish these controls.

## Kinetic models and flux

Unbiased trajectory propagation does not make a coarse transition matrix
independent of how unresolved configurations were launched within each state.
Stationary cell weights of a fitted model do not establish atomistic equilibrium
populations. Check state and lag sensitivity, within-state preparation effects,
trajectory-boundary handling, and independent data before interpreting physical
rates or stationary populations.

```bash
python scripts/validate_markov_model.py --transition transition.npy \
  --reactant 0 --product 20 --out markov_analysis.npz
```

Supply a row-stochastic matrix and the actual endpoint state indices. The tool
computes the backward committor using the time-reversed chain, reports detailed-
balance residual and the error of assuming `q_minus = 1 - q_plus`, and writes
model-derived fluxes. Passing these algebraic checks does not validate the
Markov approximation or remove adaptive sampling bias.

The local thesis source corrections are preserved in
[`publication/thesis-corrections.patch`](publication/thesis-corrections.patch).
They correct the Trp-cage segment count to 576 and qualify cell weighting and
flux claims. They do not replace the deposited university record or recompute
historical networks; those require the underlying analysis data.
The patch uses zero context; apply it to an unmodified thesis source copy with
`git apply --unidiff-zero /path/to/GESPO/docs/publication/thesis-corrections.patch`.

## Remaining release inputs

The repository has no chosen license yet; its owner must choose the license
before one is added. The exact historical benchmark artifacts, independent
replicate results, and confirmed ligand protonation/parameter files are also
pending. The example benzamidine SMILES encodes a neutral molecule; a physical
charge choice must be documented from the actual simulated system, rather than
changed silently after collecting results.

Use the pinned [local validation environment](../environments/README.md) for the
platform it names. A Linux/CUDA campaign needs its own tested environment export.
Create the final immutable release/tag and data-archive identifier after the
remaining inputs and validation are complete; this development branch is not
a claim that publication readiness has been achieved.
