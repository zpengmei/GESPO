# `gespo.projections.geom2vec` (GESPO)

Core building blocks for using **geom2vec embeddings** as the default VAMP-state backbone in GESPO.

## What’s implemented

- **Float32 file cache**: `EmbeddingCache` stores per-trajectory tensors to disk (`*.pt`) + an append-only `manifest.jsonl`.
- **Index-based lag dataset**: `TimeLaggedIndexDataset` yields `(x_t, x_{t+lag})` pairs without materializing all pairs.
- **Topology→token mapping**: `Geom2VecTopologyCache` builds residue-token maps + Cα indices from an `mdtraj.Topology`.
- **In-memory inference**: `infer_geom2vec_embeddings` runs batched inference and returns `(graph_features, ca_coords)` as CPU `float32`.
- **Pipeline helper**: `infer_and_cache` runs inference and writes to `EmbeddingCache`.

## Minimal usage

```python
import mdtraj as md
import numpy as np
from gespo.projections.geom2vec import (
    EmbeddingCache,
    Geom2VecTopologyCache,
    TimeLaggedIndexDataset,
    infer_and_cache,
)

# 1) Build tokenization (typically use a protein-only topology)
top = md.load("protein.pdb").topology
topo_cache = Geom2VecTopologyCache.from_mdtraj(top, exclude_hydrogens=True)

# 2) coords_nm: (T, atoms, 3) in nanometers (e.g., from GESPO rollouts / DCD)
coords_nm = np.load("coords_nm.npy").astype(np.float32)

# 3) Cache embeddings (returns float32, stores float32)
emb_cache = EmbeddingCache("/path/to/embeddings_cache")
rec = infer_and_cache(
    emb_cache,
    key="epoch_0001/traj_00",
    model=rep_model,
    topology=topo_cache,
    coords=coords_nm,
    stride=2,          # inference subsampling stride
    batch_size=32,
    hidden_channels=64,
    device="cuda",
)

# 4) Build an index-based lag dataset for VAMP training
graph, ca = emb_cache.load(rec.key)
ds = TimeLaggedIndexDataset(
    [{"graph_features": graph, "ca_coords": ca}],
    lag_time=10,
    pair_stride=1,
    return_index=True,
)
```

## Bootstrapping from DCDs (writes next to the DCDs)

```python
from gespo.projections.geom2vec import bootstrap_cache_from_dcd_dir

summary = bootstrap_cache_from_dcd_dir(
    model=rep_model,
    topology_file="/path/to/protein.pdb",
    trajectory_folder="/path/to/run_snaps",   # contains epoch_0001/*.dcd etc
    cache_root="/path/to/run_snaps",          # => stores *.pt alongside the DCDs
    stride=2,
    selection="protein",
    embed_batch_size=50,
    device="cuda",
    overwrite=False,
)
print("computed", len(summary["computed"]), "skipped", len(summary["skipped"]))
```

## Training the geom2vec lobe (index-based pairs)

```python
from gespo.projections.geom2vec import EmbeddingCache, train_vamp_lobe_from_cache

cache = EmbeddingCache("/path/to/run_snaps")
trained = train_vamp_lobe_from_cache(
    cache,
    keys=list(cache.keys()),
    n_states=8,
    lag_time=10,
    batch_size=256,
    n_epochs=20,
    device="cuda",
)
```

## Projector adapter

```python
from gespo.projections.geom2vec import Geom2VecVAMPProjector

projector = Geom2VecVAMPProjector(
    rep_model=rep_model,
    topology=topo_cache,
    cache=cache,
    device=torch.device("cuda"),
    hidden_channels=64,
)

projector.fit_from_cache(
    keys=list(cache.keys()),
    n_states=8,
    lag_time=10,
    n_epochs=20,
)

probs = projector.state_probs_from_cache_key("epoch_0009/epoch_0009_walker_01")
labels = projector.assign_states(probs)
```

## Using in GESPO (discrete-start loop)

The discrete-start loop (`gespo.loops._legacy.discrete`) supports a `geom2vec_vamp` projection mode that:

- infers and caches geom2vec embeddings as `float32` `*.pt` next to the per-walker `*.dcd` files under `--snapshot-dir`
- trains a geom2vec `Lobe` online with a VAMP objective
- uses the lobe output as soft states for discrete replay starts and for LOO-VAMP rewards

Minimal flags:

```bash
python -m gespo train \
  --projection geom2vec_vamp \
  --geom2vec-ckpt /path/to/geom2vec_rep.ckpt \
  --geom2vec-cache-root /path/to/run_snaps \
  --geom2vec-embed-stride 2 \
  --geom2vec-dim 16 \
  --geom2vec-lagtime 150
```

(`--tica-dim` / `--n-states` are accepted aliases of `--geom2vec-dim`; there is
no `--tica-lagtime` — use `--geom2vec-lagtime`.)
