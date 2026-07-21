# GESPO overview

GESPO (Generative Enhanced Sampling for Protein Optimization) is a reinforcement-
learning toolkit for **enhanced sampling** of protein and protein–ligand
conformational ensembles. A generative policy proposes starting structures; short
unbiased MD rollouts explore from them; the policy is rewarded for expanding the
sampled ensemble in a learned slow-coordinate (VAMP) space; and the whole loop
repeats. The result is broader, faster conformational sampling than plain MD.

## The pipeline (three stages)

```
  ┌────────────┐     ┌──────────────┐     ┌────────────┐
  │  1. seed   │ ──▶ │ 2. pretrain  │ ──▶ │  3. train  │ ──▶ iterate rounds
  │ gespo seed │     │gespo pretrain│     │ gespo train│
  └────────────┘     └──────────────┘     └────────────┘
    biased / high-T     TorsionAR policy    RL loop over
    MD trajectories     from the seeds      geom2vec-VAMP states
```

1. **Seed** ([`seeding.md`](seeding.md)) — short biased or high-temperature MD
   generates initial trajectories that cover some of the interesting motion. These
   both pretrain the policy and form the *replay pool* the RL loop resets from.
2. **Pretrain** ([`workflow.md`](workflow.md)) — `gespo pretrain` fits a **TorsionAR**
   policy (an autoregressive model over backbone — and optionally sidechain / ligand
   — torsions) to the seed trajectories. `--global-token-set` picks the collective
   variables the policy conditions on ([`reference.md`](reference.md)).
3. **Train** — `gespo train --generative-loop` runs the RL loop: it embeds
   trajectories with a **geom2vec-VAMP** projection to discover metastable states,
   rewards the policy for visiting under-sampled states, and reseeds rollouts from
   the replay pool via retrieval-steered targeted MD.

Then you **iterate**: re-pretrain the policy on the accumulated trajectories, grow
the replay pool, and run another round.

## The RL loop, one epoch at a time

`gespo train` prints per-epoch phase banners (`[Epoch e/E][Phase X]`) for the six
phases of `BaseTrainingLoop`:

| Phase | Name | What happens |
|---|---|---|
| A | Refresh projector | (re)fit the geom2vec-VAMP lobe / state clustering |
| B | Sample starts | TorsionAR proposes starts; retrieval + steering reset walkers |
| C | Run MD | short unbiased rollouts per walker |
| D | Compute rewards | LOO-VAMP, VAMP-bin entropy, MSM-info-gain, start-distance |
| E | Policy update | GRPO update of the policy from the combined reward |
| F | Log / checkpoint | stats CSV, diagnostics JSONL, plots, checkpoints |

## Two loop modes

- **generative** (`--generative-loop`, or `--mode generative`) — expand the
  ensemble around a single endpoint. The default; both examples use it.
- **path-sampling** (`--mode path-sampling`, needs `--pdb-b`) — sample transition
  paths between two endpoints A and B with committor-based rewards. See
  [`reference.md`](reference.md#path-sampling--committor).

## Where to go next

- Set up your inputs → [`inputs.md`](inputs.md)
- Generate seeds → [`seeding.md`](seeding.md)
- Full worked walkthrough → [`workflow.md`](workflow.md)
- Protein–ligand specifics → [`ligands.md`](ligands.md)
- Token sets, rewards, flags, YAML → [`reference.md`](reference.md)
- Runnable recipes → [`../examples/`](../examples/)
