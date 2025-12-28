from .._imports import *
from .rl import TorsionARPolicy
from ..utils.geometry import calculate_rmsd
from ..utils.adk_metrics import ADK_GLOBAL_TOKEN_NAMES, build_adk_token_indices, compute_adk_global_values_from_xyz
import os
from gespo.utils.logging import log_print

def _split_path_list(spec: str) -> List[str]:
    """Split a comma/semicolon/newline-separated list of paths."""
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []
    parts: List[str] = []
    for chunk in text.replace(";", ",").replace("\n", ",").split(","):
        p = str(chunk).strip()
        if p:
            parts.append(p)
    return parts

def _load_npz_dataset(path: str):
    """
    Load torsion sequences from an .npz file.

    Expected keys:
        - 'torsions': [N, L, 3] (φ, ψ, ω) in radians.
    Optional keys:
        - 'angle_mask': [N, L, 3] bool mask (True = valid angle).
        - 'residue_types': [N, L] int residue indices.
        - 'global_values': [N, G] float32 (nonnegative global scalars).
    """
    data = np.load(path)
    if "torsions" not in data:
        raise ValueError(f"{path} must contain a 'torsions' array of shape [N, L, 3].")
    torsions = data["torsions"]
    if torsions.ndim != 3 or torsions.shape[2] != 3:
        raise ValueError(f"'torsions' must have shape [N, L, 3], got {torsions.shape}.")

    angle_mask = data["angle_mask"] if "angle_mask" in data else None
    residue_types = data["residue_types"] if "residue_types" in data else None
    global_values = data["global_values"] if "global_values" in data else None
    return torsions, angle_mask, residue_types, global_values


def _build_torsion_dataset_from_dcd(
    base_dir: Union[str, List[str]],
    pdb_path: str,
    stride: int = 1,
    global_token_set: str = "rmsd_e2e",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a torsion dataset from all .dcd files under base_dir using MDTraj.

    For each frame in each trajectory:
        - Compute backbone φ, ψ, ω per residue.
        - Store as torsions[f, i, :] = (φ_i, ψ_i, ω_i).
        - Build angle_mask[f, i, a] to mark undefined angles.
        - Build residue_types[f, i] from the PDB topology.
    """
    if isinstance(base_dir, (list, tuple)):
        base_dirs = [str(p) for p in base_dir]
    else:
        base_dirs = _split_path_list(str(base_dir))
        if not base_dirs:
            base_dirs = [str(base_dir)]
    base_dirs = [os.path.abspath(p) for p in base_dirs if str(p).strip()]
    pdb_path = os.path.abspath(pdb_path)

    log_print(f"[TorsionAR Pretrain] Scanning DCDs under: {base_dirs}")
    log_print(f"[TorsionAR Pretrain] Using reference PDB: {pdb_path}")

    ref = md.load(pdb_path)
    ref_top = ref.topology
    n_res = ref_top.n_residues

    # Build residue type vocabulary via 3-letter → 1-letter mapping.
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    fallback_idx = len(aa_to_idx)  # index for unknown/other residues
    res_codes = [
        three_to_one.get(res.name.upper(), "X")
        for res in ref_top.residues
    ]
    residue_types_1d = np.array(
        [aa_to_idx.get(code, fallback_idx) for code in res_codes],
        dtype=np.int64,
    )

    def _build_residue_map(topology: md.Topology, idx_array):
        res_map = {}
        for col, atom_idx in enumerate(idx_array):
            try:
                res = topology.atom(int(atom_idx[1])).residue.index
                res_map[res] = col
            except Exception:
                continue
        return res_map

    def _per_residue_ca_indices(topology: md.Topology) -> np.ndarray:
        idx = np.full((topology.n_residues,), -1, dtype=np.int64)
        for res in topology.residues:
            for atom in res.atoms:
                if atom.name == "CA":
                    idx[res.index] = int(atom.index)
                    break
        return idx

    dcd_files: List[Tuple[str, str]] = []
    base_dir_topology: Dict[str, Optional[str]] = {}
    for bdir in base_dirs:
        sim_top = os.path.join(bdir, "simulation_topology.pdb")
        base_dir_topology[bdir] = sim_top if os.path.isfile(sim_top) else None
        for root, _, names in os.walk(bdir):
            for name in names:
                if name.endswith(".dcd"):
                    dcd_files.append((os.path.join(root, name), bdir))
    dcd_files = sorted(set(dcd_files), key=lambda x: x[0])
    if not dcd_files:
        raise RuntimeError(f"[TorsionAR Pretrain] No .dcd files found under {base_dirs}")

    torsion_blocks = []
    mask_blocks = []
    globals_blocks = []

    token_set = str(global_token_set or "rmsd_e2e").strip().lower()
    use_adk_globals = token_set in {"adk", "adk_domain", "adk_globals", "adk_thetas"}
    adk_idx = build_adk_token_indices(ref_top) if use_adk_globals else None

    total_frames = 0
    for fpath, bdir in dcd_files:
        log_print(f"[TorsionAR Pretrain] Loading {fpath} (stride={stride})")

        tr = None
        try:
            # Fast path: protein-only DCDs (implicit solvent / legacy saves).
            tr = md.load(fpath, top=pdb_path, stride=max(1, stride))
        except Exception:
            tr = None

        if tr is None:
            # Fallback: explicit-solvent runs may store full-system DCDs. Load with the
            # saved OpenMM topology and slice to protein.
            sim_top = base_dir_topology.get(bdir)
            if not sim_top:
                log_print(f"[TorsionAR Pretrain] Skipping {fpath}: no simulation_topology.pdb found under {bdir}")
                continue
            try:
                tr_full = md.load(fpath, top=sim_top, stride=max(1, stride))
                prot_idx = tr_full.topology.select("protein")
                if prot_idx is None or len(prot_idx) == 0:
                    log_print(f"[TorsionAR Pretrain] Skipping {fpath}: no protein atoms in topology {sim_top}")
                    continue
                tr = tr_full.atom_slice(prot_idx)
            except Exception as exc:
                log_print(f"[TorsionAR Pretrain] Skipping {fpath}: {exc}")
                continue

        if tr.n_frames == 0:
            continue
        if tr.topology.n_residues != n_res:
            log_print(
                f"[TorsionAR Pretrain] Skipping {fpath}: residue count mismatch "
                f"(traj n_res={tr.topology.n_residues} vs ref n_res={n_res})"
            )
            continue

        # Compute dihedral angles for this trajectory
        phi_idx, phi_vals = md.compute_phi(tr)   # [n_frames, n_phi]
        psi_idx, psi_vals = md.compute_psi(tr)   # [n_frames, n_psi]
        omg_idx, omg_vals = md.compute_omega(tr) # [n_frames, n_omega]

        phi_map = _build_residue_map(tr.topology, phi_idx)
        psi_map = _build_residue_map(tr.topology, psi_idx)
        omg_map = _build_residue_map(tr.topology, omg_idx)

        n_frames = tr.n_frames
        phi_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        psi_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        omg_arr = np.zeros((n_frames, n_res), dtype=np.float32)

        phi_mask = np.zeros((n_frames, n_res), dtype=bool)
        psi_mask = np.zeros((n_frames, n_res), dtype=bool)
        omg_mask = np.zeros((n_frames, n_res), dtype=bool)

        # Fill per-residue arrays where dihedrals are defined
        for res, col in phi_map.items():
            if col < phi_vals.shape[1]:
                phi_arr[:, res] = phi_vals[:, col].astype(np.float32)
                phi_mask[:, res] = True
        for res, col in psi_map.items():
            if col < psi_vals.shape[1]:
                psi_arr[:, res] = psi_vals[:, col].astype(np.float32)
                psi_mask[:, res] = True
        for res, col in omg_map.items():
            if col < omg_vals.shape[1]:
                omg_arr[:, res] = omg_vals[:, col].astype(np.float32)
                omg_mask[:, res] = True

        tors_block = np.stack([phi_arr, psi_arr, omg_arr], axis=-1)  # [F, L, 3]
        mask_block = np.stack([phi_mask, psi_mask, omg_mask], axis=-1)  # [F, L, 3]

        # Global metrics per frame.
        if use_adk_globals and adk_idx is not None:
            try:
                ca_idx = ref_top.select("name CA")
                if ca_idx is None or len(ca_idx) < 3:
                    glob_block = np.zeros((n_frames, len(ADK_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
                else:
                    ref_ca = ref.xyz[0, ca_idx, :].astype(np.float64)  # [n_ca,3] nm
                    glob_block = compute_adk_global_values_from_xyz(
                        tr.xyz,
                        ca_indices=np.asarray(ca_idx, dtype=np.int64),
                        ref_ca_nm=ref_ca,
                        idx=adk_idx,
                    ).astype(np.float32, copy=False)
            except Exception:
                glob_block = np.zeros((n_frames, len(ADK_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
        else:
            # Default globals: CA RMSD to reference + end-to-end CA distance.
            try:
                ca_ref = _per_residue_ca_indices(ref_top)
                ca_tr = _per_residue_ca_indices(tr.topology)
                valid = (ca_ref >= 0) & (ca_tr >= 0)
                if int(valid.sum()) < 2:
                    rmsd_nm = np.zeros((n_frames,), dtype=np.float32)
                    e2e_nm = np.zeros((n_frames,), dtype=np.float32)
                else:
                    ref_ca = ref.xyz[0, ca_ref[valid], :].astype(np.float64)  # [n_ca,3] nm
                    cur_ca = tr.xyz[:, ca_tr[valid], :].astype(np.float64)   # [F,n_ca,3] nm
                    rmsd_nm = calculate_rmsd(cur_ca, ref_ca, atom_indices=None).astype(np.float32, copy=False)
                    e2e_nm = np.linalg.norm(cur_ca[:, 0, :] - cur_ca[:, -1, :], axis=1).astype(np.float32, copy=False)
            except Exception:
                rmsd_nm = np.zeros((n_frames,), dtype=np.float32)
                e2e_nm = np.zeros((n_frames,), dtype=np.float32)
            glob_block = np.stack([rmsd_nm, e2e_nm], axis=-1).astype(np.float32, copy=False)  # [F,2]

        torsion_blocks.append(tors_block)
        mask_blocks.append(mask_block)
        globals_blocks.append(glob_block)
        total_frames += n_frames
        log_print(f"[TorsionAR Pretrain] Collected {n_frames} frames from {fpath}")

    if not torsion_blocks:
        raise RuntimeError("[TorsionAR Pretrain] No usable frames collected from DCDs.")

    torsions_all = np.concatenate(torsion_blocks, axis=0)
    mask_all = np.concatenate(mask_blocks, axis=0)
    globals_all = np.concatenate(globals_blocks, axis=0) if globals_blocks else np.zeros((torsions_all.shape[0], 0), dtype=np.float32)
    N_total = torsions_all.shape[0]
    residue_types = np.tile(residue_types_1d[None, :], (N_total, 1))

    log_print(f"[TorsionAR Pretrain] Final dataset: {torsions_all.shape[0]} frames, "
          f"{torsions_all.shape[1]} residues.")

    return torsions_all, mask_all, residue_types, globals_all


def pretrain_torsion_ar(
    source_path: str,
    out_path: str,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    kmix: int = 8,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[str] = None,
    pdb_path: Optional[str] = None,
    stride: int = 1,
    global_token_set: str = "rmsd_e2e",
) -> None:
    """
    Pretrain the TorsionARPolicy on a dataset of torsion sequences.

    Args:
        source_path:  Path to .npz file with torsion data OR directory of DCDs.
        out_path:     Where to save the pretrained weights (torch .pt).
    """
    source_dirs = _split_path_list(source_path)
    if source_dirs and all(os.path.isdir(p) for p in source_dirs):
        if not pdb_path:
            raise ValueError("pdb_path must be provided when pretraining from DCD directories.")
        torsions_np, angle_mask_np, residue_types_np, global_values_np = _build_torsion_dataset_from_dcd(
            base_dir=source_dirs,
            pdb_path=pdb_path,
            stride=stride,
            global_token_set=global_token_set,
        )
    else:
        torsions_np, angle_mask_np, residue_types_np, global_values_np = _load_npz_dataset(source_path)
    N, L, _ = torsions_np.shape
    log_print(f"[TorsionAR Pretrain] Loaded {N} sequences of length {L} from {source_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torsions = torch.from_numpy(torsions_np).float()
    if angle_mask_np is not None:
        angle_mask = torch.from_numpy(angle_mask_np.astype(np.bool_))
    else:
        angle_mask = torch.ones_like(torsions, dtype=torch.bool)

    if residue_types_np is not None:
        residue_types = torch.from_numpy(residue_types_np.astype(np.int64))
        residue_vocab = int(residue_types.max().item()) + 1
    else:
        residue_types = torch.zeros((N, L), dtype=torch.long)
        residue_vocab = 21

    token_names: Optional[Sequence[str]] = None
    if global_values_np is not None:
        global_values = torch.from_numpy(np.asarray(global_values_np, dtype=np.float32))
        if global_values.ndim != 2 or global_values.shape[0] != torsions.shape[0]:
            raise ValueError(f"'global_values' must be [N,G], got {global_values.shape}")
        n_global_tokens = int(global_values.shape[1])
        if n_global_tokens > 0:
            dataset = torch.utils.data.TensorDataset(torsions, angle_mask, residue_types, global_values)
            token_set = str(global_token_set or "").strip().lower()
            if token_set in {"adk", "adk_domain", "adk_globals", "adk_thetas"} and n_global_tokens == len(ADK_GLOBAL_TOKEN_NAMES):
                token_names = list(ADK_GLOBAL_TOKEN_NAMES)
            elif n_global_tokens == 2:
                token_names = ["rmsd_nm", "e2e_nm"]
            else:
                token_names = [f"global_{i}" for i in range(n_global_tokens)]
        else:
            dataset = torch.utils.data.TensorDataset(torsions, angle_mask, residue_types)
            n_global_tokens = 0
    else:
        dataset = torch.utils.data.TensorDataset(torsions, angle_mask, residue_types)
        n_global_tokens = 0
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = TorsionARPolicy(
        num_residues=L,
        residue_vocab_size=residue_vocab,
        d_model=d_model,
        nhead=n_heads,
        num_layers=n_layers,
        dropout=0.1,
        kmix=kmix,
        n_global_tokens=n_global_tokens,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_print(f"[TorsionAR Pretrain] Model initialized on device {device}")

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in loader:
            if n_global_tokens > 0:
                tors_b, mask_b, res_b, glob_b = batch
            else:
                tors_b, mask_b, res_b = batch
            tors_b = tors_b.to(device)
            mask_b = mask_b.to(device)
            res_b = res_b.to(device)
            if n_global_tokens > 0:
                glob_b = glob_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            logp = model.log_prob(
                tors_b,
                residue_types=res_b,
                angle_mask=mask_b,
                global_values=(glob_b if n_global_tokens > 0 else None),
            )
            loss = -logp.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * tors_b.size(0)
            total_tokens += tors_b.size(0)

        avg_loss = total_loss / max(1, total_tokens)
        log_print(f"[TorsionAR Pretrain] Epoch {ep+1}/{epochs} - NLL per sequence: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "num_residues": L,
            "residue_vocab_size": residue_vocab,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "kmix": kmix,
            "n_global_tokens": n_global_tokens,
            "global_token_names": list(token_names) if token_names is not None else None,
            "global_token_set": str(global_token_set or "").strip().lower() if n_global_tokens > 0 else "",
        },
    }
    torch.save(payload, out_path)
    log_print(f"[TorsionAR Pretrain] Saved pretrained TorsionARPolicy (with config) to {out_path}")
