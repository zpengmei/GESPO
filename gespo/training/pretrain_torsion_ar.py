import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mdtraj as md
import numpy as np
import torch
import torch.nn as nn

from .rl import TorsionARPolicy
from ..utils.geometry import calculate_rmsd
from ..utils.adk_metrics import (
    ADK_DOMAIN_TOKEN_NAMES,
    ADK_GLOBAL_TOKEN_NAMES,
    build_adk_token_indices,
    compute_adk_global_values_from_xyz,
)
from ..utils.selection import infer_protein_atom_indices, is_subset_selection
# Token-set / torsion-mode names + validators live in a dependency-light module
# (no OpenMM/torch) so the CLI and docs can reuse them; re-exported here for
# back-compat with existing importers.
from .token_sets import (  # noqa: F401
    COMMON_GLOBAL_TOKEN_SETS,
    KNOWN_GLOBAL_TOKEN_SETS,
    normalize_global_token_set,
    normalize_torsion_mode,
    validate_global_token_set,
)


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

def _load_npz_dataset(path: str, *, torsion_mode: str = "backbone"):
    """
    Load torsion sequences from an .npz file.

    Expected keys:
        - 'torsions': [N, L, A] torsion angles in radians.
    Optional keys:
        - 'angle_mask': [N, L, A] bool mask (True = valid angle).
        - 'residue_types': [N, L] int residue indices.
        - 'global_values': [N, G] float32 (nonnegative global scalars).
    """
    data = np.load(path)
    if "torsions" not in data:
        raise ValueError(f"{path} must contain a 'torsions' array of shape [N, L, A].")
    torsions = data["torsions"]
    if torsions.ndim != 3 or torsions.shape[2] < 3:
        raise ValueError(f"'torsions' must have shape [N, L, A>=3], got {torsions.shape}.")

    angle_mask = data["angle_mask"] if "angle_mask" in data else None
    if angle_mask is not None and np.asarray(angle_mask).shape != torsions.shape:
        raise ValueError(f"'angle_mask' must have the same shape as 'torsions' {torsions.shape}, got {np.shape(angle_mask)}.")

    mode = str(torsion_mode or "backbone").strip().lower()
    if mode in {"backbone", "backbone_only", "backbone-only"}:
        # Allow loading larger torsion tensors but slice down to backbone-only.
        if int(torsions.shape[2]) > 3:
            torsions = torsions[:, :, :3]
            if angle_mask is not None:
                angle_mask = np.asarray(angle_mask)[:, :, :3]
    elif mode in {"backbone+sidechain", "backbone_sidechain", "backbone+sc", "bb+sc"}:
        if int(torsions.shape[2]) <= 3:
            raise ValueError(
                f"[TorsionAR Pretrain] torsion_mode='{mode}' requires A>3 (sidechain chis), got torsions {torsions.shape}."
            )
    else:
        raise ValueError(
            f"[TorsionAR Pretrain] Unknown torsion_mode='{mode}'. Expected 'backbone' or 'backbone+sidechain'."
        )
    residue_types = data["residue_types"] if "residue_types" in data else None
    global_values = data["global_values"] if "global_values" in data else None
    return torsions, angle_mask, residue_types, global_values


def _build_torsion_dataset_from_dcd(
    base_dir: Union[str, List[str]],
    pdb_path: str,
    pdb_b_path: Optional[str] = None,
    stride: int = 1,
    global_token_set: str = "rmsd_e2e",
    torsion_mode: str = "backbone",
    ligand_smiles: str = "",
    pocket_ref_pdb: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a torsion dataset from trajectory files under base_dir using MDTraj.

    For each frame in each trajectory, we build a fixed-width per-residue torsion tensor.

    Torsion modes:
        - backbone: φ, ψ, ω
        - backbone+sidechain: φ, ψ, ω, χ1..χ4 (when available; masked otherwise)

    Layout (when backbone+sidechain):
        torsions[f, i, :] = (φ_i, ψ_i, ω_i, χ1_i, χ2_i, χ3_i, χ4_i).
        - Build angle_mask[f, i, a] to mark undefined angles.
        - Build residue_types[f, i] from the PDB topology.
    """
    mode = str(torsion_mode or "backbone").strip().lower()
    include_sidechains = mode in {"backbone+sidechain", "backbone_sidechain", "backbone+sc", "bb+sc"}
    if not include_sidechains and mode not in {"backbone", "backbone_only", "backbone-only"}:
        raise ValueError(
            f"[TorsionAR Pretrain] Unknown torsion_mode='{mode}'. Expected 'backbone' or 'backbone+sidechain'."
        )

    if isinstance(base_dir, (list, tuple)):
        base_dirs = [str(p) for p in base_dir]
    else:
        base_dirs = _split_path_list(str(base_dir))
        if not base_dirs:
            base_dirs = [str(base_dir)]
    base_dirs = [os.path.abspath(p) for p in base_dirs if str(p).strip()]
    pdb_path = os.path.abspath(pdb_path)

    print(f"[TorsionAR Pretrain] Scanning trajectories under: {base_dirs}")
    print(f"[TorsionAR Pretrain] Using reference PDB: {pdb_path}")

    ref_full = md.load(pdb_path)
    atom_selection = None
    try:
        prot_idx = infer_protein_atom_indices(ref_full.topology, include_caps=True, allow_empty=True)
        if is_subset_selection(prot_idx, int(ref_full.n_atoms)):
            atom_selection = np.asarray(prot_idx, dtype=np.int64)
            ref = ref_full.atom_slice(atom_selection)
            print(
                f"[TorsionAR Pretrain] Detected non-protein residues in PDB; using protein-like atom subset "
                f"({len(atom_selection)}/{ref_full.n_atoms} atoms)."
            )
        else:
            ref = ref_full
    except Exception:
        ref = ref_full
    ref_top = ref.topology
    # Filter out non-amino-acid residues that infer_protein_atom_indices may have
    # incorrectly included (e.g., UNK ligand can be included if it has standard
    # backbone-like atom names).
    PROTEIN_AA = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP",
                  "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","ACE","NME"}
    _non_aa = [r for r in ref_top.residues if r.name.upper() not in PROTEIN_AA]
    if _non_aa:
        _aa_atoms = [a.index for r in ref_top.residues if r.name.upper() in PROTEIN_AA for a in r.atoms]
        if _aa_atoms:
            print(f"[TorsionAR Pretrain] Filtering {len(_non_aa)} non-AA residue(s) from protein "
                  f"topology: {[r.name for r in _non_aa]}")
            ref = ref.atom_slice(_aa_atoms)
            ref_top = ref.topology
    n_res = ref_top.n_residues

    # Detect ligand rotatable dihedrals in the full (unsplit) topology.
    # Use SMILES-based RDKit detection if available (allows H neighbors, e.g. amidine).
    _lig_smiles = str(ligand_smiles or "")
    if _lig_smiles:
        from gespo.utils.geometry import find_ligand_rotatable_dihedrals_smiles
        ligand_dihedrals = find_ligand_rotatable_dihedrals_smiles(ref_full.topology, _lig_smiles)
        print(f"[TorsionAR Pretrain] Using RDKit+SMILES rotatable bond detection (smiles={_lig_smiles})")
    else:
        from gespo.utils.geometry import find_ligand_rotatable_dihedrals
        ligand_dihedrals = find_ligand_rotatable_dihedrals(ref_full.topology)
    n_ligand_torsions = len(ligand_dihedrals)
    if n_ligand_torsions > 0:
        print(f"[TorsionAR Pretrain] Found {n_ligand_torsions} ligand rotatable dihedral(s):")
        for label, (i, j, k, l) in ligand_dihedrals:
            print(f"  {label}: atoms ({i},{j},{k},{l})")
        # NEW LAYOUT: each ligand dihedral gets its OWN position in the sequence
        # (instead of cramming all into 1 slot at different angle indices).
        n_res_total = n_res + n_ligand_torsions
    else:
        n_res_total = n_res

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

    def _infer_sim_topology_path(bdir: str) -> Optional[str]:
        """
        Try to locate a solvated simulation topology PDB that matches full-system DCDs.

        Supports common layouts:
          - <base>/simulation_topology.pdb                  (run_snaps roots)
          - <base>/solvation/simulation_topology.pdb        (bootstrap/direct-MD roots)
          - <base>/../solvation/simulation_topology.pdb     (direct-MD run_A/run_B dirs)
        """
        bdir = os.path.abspath(str(bdir))
        parent = os.path.abspath(os.path.join(bdir, os.pardir))
        candidates = [
            os.path.join(bdir, "simulation_topology.pdb"),
            os.path.join(bdir, "run_snaps", "simulation_topology.pdb"),
            os.path.join(bdir, "solvation", "simulation_topology.pdb"),
            os.path.join(parent, "simulation_topology.pdb"),
            os.path.join(parent, "solvation", "simulation_topology.pdb"),
        ]
        for p in candidates:
            try:
                if p and os.path.isfile(p):
                    return p
            except Exception:
                continue
        return None

    traj_files: List[Tuple[str, str]] = []
    base_dir_topology: Dict[str, Optional[str]] = {}
    for bdir in base_dirs:
        base_dir_topology[bdir] = _infer_sim_topology_path(bdir)
        for root, _, names in os.walk(bdir):
            for name in names:
                low = str(name).lower()
                if low.endswith(".dcd") or low.endswith(".xtc"):
                    traj_files.append((os.path.join(root, name), bdir))
    traj_files = sorted(set(traj_files), key=lambda x: x[0])
    if not traj_files:
        raise RuntimeError(f"[TorsionAR Pretrain] No .dcd/.xtc files found under {base_dirs}")

    first_traj_for_pocket_log = True
    _AA_SET = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP",
               "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","ACE","NME"}
    torsion_blocks = []
    mask_blocks = []
    globals_blocks = []

    token_set = validate_global_token_set(global_token_set)
    token_set_compact = token_set.replace(" ", "")
    use_adk_domain_only = token_set in {
        "adk_domain4",
        "adk_domain_only",
        "adk_domain_no_rmsd",
        "adk_no_rmsd",
    }
    use_adk_domain4_rmsd_ab = token_set in {
        "adk_domain4_rmsd_ab",
        "adk_domain4_rmsd_a_b",
        "adk_domain4_rmsd_ab_nm",
        "adk_domain4_rmsd_a_b_nm",
    }
    use_adk_globals = use_adk_domain_only or token_set in {"adk", "adk_domain", "adk_globals", "adk_thetas"}
    use_rg = token_set in {"rmsd_e2e_rg", "rmsd_e2e_rg_nm", "rg", "rg_nm"}
    use_rmsd_ab = token_set in {"rmsd_ab", "rmsd_a_b", "rmsd_ab_nm", "rmsd_a_b_nm"} or use_adk_domain4_rmsd_ab
    use_rmsd_ab_e2e = token_set in {
        "rmsd_ab_e2e",
        "rmsd_a_b_e2e",
        "rmsd_ab_e2e_nm",
        "rmsd_a_b_e2e_nm",
        "path_rmsd_ab",
        "path_rmsd_ab_e2e",
    }
    use_adk_globals = use_adk_globals or use_adk_domain4_rmsd_ab
    use_vinculin_h5 = token_set in {"vinculin_h5", "vinculin_h5_bundle", "h5_bundle"}
    use_ppi_globals = token_set_compact in {
        "ppi",
        "ppi_global",
        "ppi_globals",
        "ppi_tokens",
        "chain_com_dist_nm,min_interface_ca_dist_nm,q_interface",
    }

    def _select_ca_indices(top) -> np.ndarray:
        """Prefer protein CA selection, but fall back to name-only if topology lacks protein flags."""
        ca_idx = None
        try:
            ca_idx = top.select("protein and name CA")
        except Exception:
            ca_idx = None
        if ca_idx is None or len(ca_idx) == 0:
            try:
                ca_idx = top.select("name CA")
            except Exception:
                ca_idx = None
        return np.asarray(ca_idx if ca_idx is not None else [], dtype=np.int64).reshape(-1)

    def _build_adk_idx_best(top):
        """
        Build ADK token indices against the *trajectory topology* used to index `tr.xyz`.

        Some topologies have unreliable/absent `resSeq` labels; try both conventions
        and pick the one that yields the richest set of group atoms/pairs.
        """
        best = None
        best_score = -1
        for prefer_resseq in (True, False):
            try:
                idx = build_adk_token_indices(top, protein_indices=None, prefer_resseq=prefer_resseq)
            except Exception:
                continue
            try:
                group_score = int(sum(int(v.size) for v in idx.group_atoms.values()))
            except Exception:
                group_score = 0
            try:
                pair_score = int(len(getattr(idx, "pairs", {}) or {}))
            except Exception:
                pair_score = 0
            score = group_score + 1000 * pair_score
            if score > best_score:
                best_score = score
                best = idx
        return best
    ref_b = None
    ref_top_b = None
    if (use_rmsd_ab or use_rmsd_ab_e2e) and pdb_b_path:
        try:
            ref_b_full = md.load(pdb_b_path)
            try:
                prot_idx_b = infer_protein_atom_indices(ref_b_full.topology, include_caps=True, allow_empty=True)
                if is_subset_selection(prot_idx_b, int(ref_b_full.n_atoms)):
                    ref_b = ref_b_full.atom_slice(np.asarray(prot_idx_b, dtype=np.int64))
                else:
                    ref_b = ref_b_full
            except Exception:
                ref_b = ref_b_full
            ref_top_b = ref_b.topology if ref_b is not None else None
            if ref_b.n_frames <= 0:
                ref_b = None
                ref_top_b = None
        except Exception:
            ref_b = None
            ref_top_b = None
    if (use_rmsd_ab or use_rmsd_ab_e2e) and (ref_b is None or ref_top_b is None):
        raise ValueError("global_token_set requires a valid pdb_b_path for RMSD-to-B (e.g. global_token_set='rmsd_ab_e2e').")
    if (use_rmsd_ab or use_rmsd_ab_e2e) and ref_top_b is not None and int(ref_top_b.n_residues) != int(n_res):
        print(
            f"[TorsionAR Pretrain] Warning: endpoint B residue count ({int(ref_top_b.n_residues)}) "
            f"!= endpoint A residue count ({int(n_res)}). RMSD-to-B tokens may be unreliable.",
            flush=True,
        )

    total_frames = 0
    for fpath, bdir in traj_files:
        print(f"[TorsionAR Pretrain] Loading {fpath} (stride={stride})")

        tr = None
        _tr_full_for_ligand = None
        try:
            # Fast path: protein-only DCDs (implicit solvent / legacy saves).
            tr = md.load(fpath, top=ref_top, stride=max(1, stride))
        except Exception:
            tr = None

        if tr is None:
            # Fallback: explicit-solvent runs may store full-system DCDs. Load with the
            # saved OpenMM topology and slice to protein.
            sim_top = base_dir_topology.get(bdir)
            if not sim_top:
                print(
                    f"[TorsionAR Pretrain] Skipping {fpath}: could not locate a solvated "
                    f"simulation_topology.pdb for base_dir={bdir} (needed for full-system DCDs)."
                )
                continue
            try:
                tr_full = md.load(fpath, top=sim_top, stride=max(1, stride))
                prot_idx = tr_full.topology.select("protein")
                if prot_idx is None or len(prot_idx) == 0:
                    prot_idx = infer_protein_atom_indices(tr_full.topology, include_caps=True, allow_empty=True)
                if prot_idx is None or len(prot_idx) == 0:
                    print(f"[TorsionAR Pretrain] Skipping {fpath}: no protein atoms in topology {sim_top}")
                    continue
                # Keep full trajectory for ligand torsion extraction later
                _tr_full_for_ligand = tr_full
                tr = tr_full.atom_slice(prot_idx)
                # Filter UNK/non-AA residues (consistent with reference topology filtering).
                _aa_atoms_in_tr = [a.index for r in tr.topology.residues
                                   if r.name.upper() in PROTEIN_AA for a in r.atoms]
                if _aa_atoms_in_tr and len(_aa_atoms_in_tr) < tr.n_atoms:
                    tr = tr.atom_slice(_aa_atoms_in_tr)
            except Exception as exc:
                print(f"[TorsionAR Pretrain] Skipping {fpath}: {exc}")
                continue

        if tr.n_frames == 0:
            continue
        if tr.topology.n_residues != n_res:
            print(
                f"[TorsionAR Pretrain] Skipping {fpath}: residue count mismatch "
                f"(traj n_res={tr.topology.n_residues} vs ref n_res={n_res})"
            )
            continue

        # Compute dihedral angles for this trajectory.
        phi_idx, phi_vals = md.compute_phi(tr)   # [n_frames, n_phi]
        psi_idx, psi_vals = md.compute_psi(tr)   # [n_frames, n_psi]
        omg_idx, omg_vals = md.compute_omega(tr) # [n_frames, n_omega]
        if include_sidechains:
            try:
                chi1_idx, chi1_vals = md.compute_chi1(tr)
            except Exception:
                chi1_idx, chi1_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)
            try:
                chi2_idx, chi2_vals = md.compute_chi2(tr)
            except Exception:
                chi2_idx, chi2_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)
            try:
                chi3_idx, chi3_vals = md.compute_chi3(tr)
            except Exception:
                chi3_idx, chi3_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)
            try:
                chi4_idx, chi4_vals = md.compute_chi4(tr)
            except Exception:
                chi4_idx, chi4_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)
        else:
            chi1_idx, chi1_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)
            chi2_idx, chi2_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)
            chi3_idx, chi3_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)
            chi4_idx, chi4_vals = np.zeros((0, 4), dtype=np.int64), np.zeros((tr.n_frames, 0), dtype=np.float32)

        phi_map = _build_residue_map(tr.topology, phi_idx)
        psi_map = _build_residue_map(tr.topology, psi_idx)
        omg_map = _build_residue_map(tr.topology, omg_idx)
        chi1_map = _build_residue_map(tr.topology, chi1_idx)
        chi2_map = _build_residue_map(tr.topology, chi2_idx)
        chi3_map = _build_residue_map(tr.topology, chi3_idx)
        chi4_map = _build_residue_map(tr.topology, chi4_idx)

        n_frames = tr.n_frames
        phi_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        psi_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        omg_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        chi1_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        chi2_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        chi3_arr = np.zeros((n_frames, n_res), dtype=np.float32)
        chi4_arr = np.zeros((n_frames, n_res), dtype=np.float32)

        phi_mask = np.zeros((n_frames, n_res), dtype=bool)
        psi_mask = np.zeros((n_frames, n_res), dtype=bool)
        omg_mask = np.zeros((n_frames, n_res), dtype=bool)
        chi1_mask = np.zeros((n_frames, n_res), dtype=bool)
        chi2_mask = np.zeros((n_frames, n_res), dtype=bool)
        chi3_mask = np.zeros((n_frames, n_res), dtype=bool)
        chi4_mask = np.zeros((n_frames, n_res), dtype=bool)

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
        for res, col in chi1_map.items():
            if col < chi1_vals.shape[1]:
                chi1_arr[:, res] = chi1_vals[:, col].astype(np.float32)
                chi1_mask[:, res] = True
        for res, col in chi2_map.items():
            if col < chi2_vals.shape[1]:
                chi2_arr[:, res] = chi2_vals[:, col].astype(np.float32)
                chi2_mask[:, res] = True
        for res, col in chi3_map.items():
            if col < chi3_vals.shape[1]:
                chi3_arr[:, res] = chi3_vals[:, col].astype(np.float32)
                chi3_mask[:, res] = True
        for res, col in chi4_map.items():
            if col < chi4_vals.shape[1]:
                chi4_arr[:, res] = chi4_vals[:, col].astype(np.float32)
                chi4_mask[:, res] = True

        if include_sidechains:
            tors_block = np.stack([phi_arr, psi_arr, omg_arr, chi1_arr, chi2_arr, chi3_arr, chi4_arr], axis=-1)  # [F,L,7]
            mask_block = np.stack([phi_mask, psi_mask, omg_mask, chi1_mask, chi2_mask, chi3_mask, chi4_mask], axis=-1)  # [F,L,7]
        else:
            tors_block = np.stack([phi_arr, psi_arr, omg_arr], axis=-1)  # [F,L,3]
            mask_block = np.stack([phi_mask, psi_mask, omg_mask], axis=-1)  # [F,L,3]

        # Append ligand rotatable dihedrals — ONE POSITION PER DIHEDRAL.
        # Layout: each ligand dihedral occupies its own residue position in the
        # sequence (using the SAME torsion representation as backbone). The dihedral
        # value is stored at angle index 0 (phi-like); other angles are masked off.
        if n_ligand_torsions > 0:
            n_a = tors_block.shape[2]  # angles per residue
            lig_tors = np.zeros((n_frames, n_ligand_torsions, n_a), dtype=np.float32)
            lig_mask = np.zeros((n_frames, n_ligand_torsions, n_a), dtype=bool)

            # Compute ligand dihedrals from the full trajectory (before protein slicing).
            lig_source = _tr_full_for_ligand
            if lig_source is None:
                # Protein-only DCD — no ligand atoms available; use zeros + masked out
                pass
            else:
                for d_idx, (label, (ai, aj, ak, al)) in enumerate(ligand_dihedrals):
                    try:
                        dih_indices = np.array([[ai, aj, ak, al]], dtype=np.int32)
                        dih_vals = md.compute_dihedrals(lig_source, dih_indices)  # [F, 1]
                        # NEW: store at d_idx-th position (own row), angle column 0.
                        lig_tors[:, d_idx, 0] = dih_vals[:, 0].astype(np.float32)
                        lig_mask[:, d_idx, 0] = True
                    except Exception as exc_lig:
                        print(f"[TorsionAR Pretrain] Warning: failed to compute ligand dihedral {label}: {exc_lig}")

            tors_block = np.concatenate([tors_block, lig_tors], axis=1)  # [F, L+N_lig, A]
            mask_block = np.concatenate([mask_block, lig_mask], axis=1)  # [F, L+N_lig, A]

        # Global metrics per frame.
        if use_adk_domain4_rmsd_ab:
            # (rmsd_a_nm, rmsd_b_nm, theta_nmp_rad, theta_lid_rad, dist_145_52_nm, dist_169_55_nm)
            # RMSD_A/B uses per-residue CA matching; domain features use the ADK group definitions.
            try:
                ca_ref = _per_residue_ca_indices(ref_top)
                ca_tr = _per_residue_ca_indices(tr.topology)
                if ref_b is None or ref_top_b is None:
                    raise ValueError("adk_domain4_rmsd_ab requires pdb_b_path")
                ca_ref_b = _per_residue_ca_indices(ref_top_b)
                valid = (ca_ref >= 0) & (ca_ref_b >= 0) & (ca_tr >= 0)
                if int(valid.sum()) < 2:
                    rmsd_a_nm = np.zeros((n_frames,), dtype=np.float32)
                    rmsd_b_nm = np.zeros((n_frames,), dtype=np.float32)
                else:
                    ref_a_ca = ref.xyz[0, ca_ref[valid], :].astype(np.float64)  # [n_ca,3] nm
                    ref_b_ca = ref_b.xyz[0, ca_ref_b[valid], :].astype(np.float64)  # [n_ca,3] nm
                    cur_ca = tr.xyz[:, ca_tr[valid], :].astype(np.float64)  # [F,n_ca,3] nm
                    rmsd_a_nm = calculate_rmsd(cur_ca, ref_a_ca, atom_indices=None).astype(np.float32, copy=False)
                    rmsd_b_nm = calculate_rmsd(cur_ca, ref_b_ca, atom_indices=None).astype(np.float32, copy=False)
            except Exception:
                rmsd_a_nm = np.zeros((n_frames,), dtype=np.float32)
                rmsd_b_nm = np.zeros((n_frames,), dtype=np.float32)

            dom4 = np.zeros((n_frames, len(ADK_DOMAIN_TOKEN_NAMES)), dtype=np.float32)
            try:
                # Build ADK indices in the trajectory topology (aligned to `tr.xyz`).
                adk_idx_tr = _build_adk_idx_best(tr.topology) if use_adk_globals else None
                if adk_idx_tr is not None:
                    ca_idx = _select_ca_indices(tr.topology)
                    if int(ca_idx.size) >= 3:
                        # Domain features do not depend on the RMSD column; provide a consistent
                        # CA reference in the same ordering as `ca_idx`.
                        ref_ca = tr.xyz[0, ca_idx, :].astype(np.float64)  # [n_ca,3] nm
                        full5 = compute_adk_global_values_from_xyz(
                            tr.xyz,
                            ca_indices=np.asarray(ca_idx, dtype=np.int64),
                            ref_ca_nm=ref_ca,
                            idx=adk_idx_tr,
                        ).astype(np.float32, copy=False)
                        if full5.ndim == 2 and int(full5.shape[1]) >= 1 + len(ADK_DOMAIN_TOKEN_NAMES):
                            dom4 = np.asarray(full5[:, 1 : 1 + len(ADK_DOMAIN_TOKEN_NAMES)], dtype=np.float32)
                    # Warn if we still got all-zeros domain tokens (usually indicates residue-number mapping failure).
                    if not np.any(dom4):
                        try:
                            resseqs = []
                            for r in tr.topology.residues:
                                try:
                                    resseqs.append(int(getattr(r, "resSeq", r.index)))
                                except Exception:
                                    resseqs.append(int(r.index))
                            msg = (
                                f"[TorsionAR Pretrain] Warning: ADK domain globals are all zeros for {os.path.basename(fpath)}; "
                                f"top_residues={int(tr.topology.n_residues)} resSeq_range=[{min(resseqs)},{max(resseqs)}] "
                                f"ca={int(ca_idx.size)} pairs={len(getattr(adk_idx_tr,'pairs',{}) or {})}. "
                                "Check that trajectories are ADK and residue numbering matches ADK definitions."
                            )
                            print(msg, flush=True)
                        except Exception:
                            pass
            except Exception as exc:
                try:
                    print(
                        f"[TorsionAR Pretrain] Warning: failed to compute ADK domain globals for {os.path.basename(fpath)} ({exc})",
                        flush=True,
                    )
                except Exception:
                    pass
                dom4 = np.zeros((n_frames, len(ADK_DOMAIN_TOKEN_NAMES)), dtype=np.float32)

            glob_block = np.concatenate(
                [
                    np.asarray(rmsd_a_nm, dtype=np.float32).reshape(-1, 1),
                    np.asarray(rmsd_b_nm, dtype=np.float32).reshape(-1, 1),
                    np.asarray(dom4, dtype=np.float32),
                ],
                axis=1,
            )
        elif use_adk_globals:
            try:
                # Use protein CA only (avoid Ca2+ ions named "CA" in explicit-solvent topologies).
                ca_idx = _select_ca_indices(tr.topology)
                if int(ca_idx.size) < 3:
                    n_cols = len(ADK_DOMAIN_TOKEN_NAMES) if use_adk_domain_only else len(ADK_GLOBAL_TOKEN_NAMES)
                    glob_block = np.zeros((n_frames, n_cols), dtype=np.float32)
                else:
                    # Build ADK indices in the trajectory topology (aligned to `tr.xyz`).
                    adk_idx_tr = _build_adk_idx_best(tr.topology)
                    if adk_idx_tr is None:
                        raise RuntimeError("failed to build ADK token indices for trajectory topology")
                    # Align endpoint-A CA reference to the trajectory CA ordering by residue resSeq.
                    # This avoids subtle mismatches when the trajectory topology reorders atoms.
                    ref_ca = None
                    try:
                        ref_ca_by_resseq = {}
                        for atom in ref_top.atoms:
                            if str(getattr(atom, "name", "")) != "CA":
                                continue
                            res = atom.residue
                            try:
                                rs = int(getattr(res, "resSeq", res.index))
                            except Exception:
                                rs = int(res.index)
                            ref_ca_by_resseq[rs] = ref.xyz[0, int(atom.index), :].astype(np.float64)
                        tr_ca = []
                        for ai in np.asarray(ca_idx, dtype=np.int64).reshape(-1).tolist():
                            atom = tr.topology.atom(int(ai))
                            try:
                                rs = int(getattr(atom.residue, "resSeq", atom.residue.index))
                            except Exception:
                                rs = int(atom.residue.index)
                            tr_ca.append(ref_ca_by_resseq.get(rs, tr.xyz[0, int(ai), :].astype(np.float64)))
                        ref_ca = np.asarray(tr_ca, dtype=np.float64).reshape(-1, 3)
                    except Exception:
                        ref_ca = tr.xyz[0, ca_idx, :].astype(np.float64)
                    glob_block = compute_adk_global_values_from_xyz(
                        tr.xyz,
                        ca_indices=np.asarray(ca_idx, dtype=np.int64),
                        ref_ca_nm=ref_ca,
                        idx=adk_idx_tr,
                    ).astype(np.float32, copy=False)
                    if use_adk_domain_only and glob_block.ndim == 2 and int(glob_block.shape[1]) >= 1 + len(ADK_DOMAIN_TOKEN_NAMES):
                        glob_block = np.asarray(glob_block[:, 1 : 1 + len(ADK_DOMAIN_TOKEN_NAMES)], dtype=np.float32)
            except Exception:
                n_cols = len(ADK_DOMAIN_TOKEN_NAMES) if use_adk_domain_only else len(ADK_GLOBAL_TOKEN_NAMES)
                glob_block = np.zeros((n_frames, n_cols), dtype=np.float32)
        elif use_ppi_globals:
            try:
                from gespo.utils.ppi_metrics import (
                    PPI_GLOBAL_TOKEN_NAMES,
                    build_ppi_token_indices,
                    compute_ppi_globals_from_xyz,
                    set_native_contacts as set_ppi_native_contacts,
                )

                ppi_idx = build_ppi_token_indices(
                    tr.topology,
                    protein_indices=np.arange(int(tr.n_atoms), dtype=np.int64),
                    chain_a_id=0,
                    chain_b_id=1,
                )
                ref_xyz = ref.xyz[0].astype(np.float64)
                ppi_idx = set_ppi_native_contacts(ppi_idx, ref_xyz, cutoff_nm=0.8)
                glob_block = compute_ppi_globals_from_xyz(tr.xyz, ppi_idx).astype(np.float32, copy=False)
                if glob_block.ndim != 2 or int(glob_block.shape[1]) != len(PPI_GLOBAL_TOKEN_NAMES):
                    raise ValueError(f"unexpected PPI globals shape {glob_block.shape}")
            except Exception as exc:
                print(f"[TorsionAR Pretrain] Warning: PPI globals failed for {os.path.basename(fpath)}: {exc}")
                from gespo.utils.ppi_metrics import PPI_GLOBAL_TOKEN_NAMES
                glob_block = np.zeros((n_frames, len(PPI_GLOBAL_TOKEN_NAMES)), dtype=np.float32)
        elif use_vinculin_h5:
            # Vinculin H5-focused globals: h5_bundle_dist, h5_h1_angle, linker_e2e, Q_h5, h5_helicity
            try:
                from gespo.utils.vinculin_h5_metrics import (
                    compute_vinculin_h5_globals,
                    set_native_contacts_from_ref,
                    VINCULIN_H5_TOKEN_NAMES,
                )
                ca_ref_arr = _per_residue_ca_indices(ref_top)
                ca_tr_arr = _per_residue_ca_indices(tr.topology)
                valid = (ca_ref_arr >= 0) & (ca_tr_arr >= 0)
                if int(valid.sum()) >= 180:
                    # Set native contacts from reference (idempotent)
                    ref_ca_180 = ref.xyz[0, ca_ref_arr[valid][:180], :].astype(np.float64)
                    set_native_contacts_from_ref(ref_ca_180, cutoff_nm=0.8)
                    # Extract CA coords: (n_frames, 180, 3)
                    ca_coords = tr.xyz[:, ca_tr_arr[valid][:180], :].astype(np.float64)
                    glob_block = compute_vinculin_h5_globals(ca_coords)
                else:
                    glob_block = np.zeros((n_frames, len(VINCULIN_H5_TOKEN_NAMES)), dtype=np.float32)
            except Exception as exc:
                print(f"[TorsionAR Pretrain] Warning: vinculin_h5 globals failed for {os.path.basename(fpath)}: {exc}")
                from gespo.utils.vinculin_h5_metrics import VINCULIN_H5_TOKEN_NAMES
                glob_block = np.zeros((n_frames, len(VINCULIN_H5_TOKEN_NAMES)), dtype=np.float32)
        else:
            # Default globals: CA RMSD to reference + end-to-end CA distance (+ optional Rg).
            try:
                ca_ref = _per_residue_ca_indices(ref_top)
                ca_tr = _per_residue_ca_indices(tr.topology)
                if use_rmsd_ab or use_rmsd_ab_e2e:
                    if ref_b is None or ref_top_b is None:
                        raise ValueError("rmsd_ab token set requires pdb_b_path")
                    ca_ref_b = _per_residue_ca_indices(ref_top_b)
                    valid = (ca_ref >= 0) & (ca_ref_b >= 0) & (ca_tr >= 0)
                    if int(valid.sum()) < 2:
                        rmsd_a_nm = np.zeros((n_frames,), dtype=np.float32)
                        rmsd_b_nm = np.zeros((n_frames,), dtype=np.float32)
                        e2e_nm = np.zeros((n_frames,), dtype=np.float32) if use_rmsd_ab_e2e else None
                    else:
                        ref_a_ca = ref.xyz[0, ca_ref[valid], :].astype(np.float64)  # [n_ca,3] nm
                        ref_b_ca = ref_b.xyz[0, ca_ref_b[valid], :].astype(np.float64)  # [n_ca,3] nm
                        cur_ca = tr.xyz[:, ca_tr[valid], :].astype(np.float64)  # [F,n_ca,3] nm
                        rmsd_a_nm = calculate_rmsd(cur_ca, ref_a_ca, atom_indices=None).astype(np.float32, copy=False)
                        rmsd_b_nm = calculate_rmsd(cur_ca, ref_b_ca, atom_indices=None).astype(np.float32, copy=False)
                        if use_rmsd_ab_e2e:
                            e2e_nm = np.linalg.norm(cur_ca[:, 0, :] - cur_ca[:, -1, :], axis=1).astype(
                                np.float32, copy=False
                            )
                        else:
                            e2e_nm = None
                else:
                    valid = (ca_ref >= 0) & (ca_tr >= 0)
                    if int(valid.sum()) < 2:
                        rmsd_nm = np.zeros((n_frames,), dtype=np.float32)
                        e2e_nm = np.zeros((n_frames,), dtype=np.float32)
                        rg_nm = np.zeros((n_frames,), dtype=np.float32) if use_rg else None
                    else:
                        ref_ca = ref.xyz[0, ca_ref[valid], :].astype(np.float64)  # [n_ca,3] nm
                        cur_ca = tr.xyz[:, ca_tr[valid], :].astype(np.float64)  # [F,n_ca,3] nm
                        rmsd_nm = calculate_rmsd(cur_ca, ref_ca, atom_indices=None).astype(np.float32, copy=False)
                        e2e_nm = np.linalg.norm(cur_ca[:, 0, :] - cur_ca[:, -1, :], axis=1).astype(
                            np.float32, copy=False
                        )
                        if use_rg:
                            centered = cur_ca - cur_ca.mean(axis=1, keepdims=True)
                            rg_nm = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1)).astype(
                                np.float32, copy=False
                            )
                        else:
                            rg_nm = None
            except Exception:
                if use_rmsd_ab or use_rmsd_ab_e2e:
                    rmsd_a_nm = np.zeros((n_frames,), dtype=np.float32)
                    rmsd_b_nm = np.zeros((n_frames,), dtype=np.float32)
                    e2e_nm = np.zeros((n_frames,), dtype=np.float32) if use_rmsd_ab_e2e else None
                else:
                    rmsd_nm = np.zeros((n_frames,), dtype=np.float32)
                    e2e_nm = np.zeros((n_frames,), dtype=np.float32)
                    rg_nm = np.zeros((n_frames,), dtype=np.float32) if use_rg else None

            if use_rmsd_ab_e2e:
                if e2e_nm is None:
                    e2e_nm = np.zeros((n_frames,), dtype=np.float32)
                glob_block = np.stack([rmsd_a_nm, rmsd_b_nm, e2e_nm], axis=-1).astype(np.float32, copy=False)  # [F,3]
            elif use_rmsd_ab:
                glob_block = np.stack([rmsd_a_nm, rmsd_b_nm], axis=-1).astype(np.float32, copy=False)  # [F,2]
            elif use_rg:
                if rg_nm is None:
                    rg_nm = np.zeros((n_frames,), dtype=np.float32)
                glob_block = np.stack([rmsd_nm, e2e_nm, rg_nm], axis=-1).astype(np.float32, copy=False)  # [F,3]
            else:
                glob_block = np.stack([rmsd_nm, e2e_nm], axis=-1).astype(np.float32, copy=False)  # [F,2]

            # Append ligand-protein distance if requested.
            if token_set in {"rmsd_e2e_ligdist", "rmsd_e2e_lig"}:
                try:
                    lig_heavy = np.array([a.index for a in tr.topology.atoms
                                          if a.residue.name in ("UNK", "BEN", "LIG") and a.element.symbol != "H"], dtype=int)
                    asp_heavy = np.array([a.index for a in tr.topology.atoms
                                          if a.residue.resSeq == 189 and a.element.symbol != "H"
                                          and a.residue.name in ("ASP", "ASN")], dtype=int)
                    if len(lig_heavy) > 0 and len(asp_heavy) > 0:
                        pairs = np.array([(i, j) for i in lig_heavy for j in asp_heavy])
                        dists = md.compute_distances(tr, pairs)  # [F, n_pairs] in nm
                        lig_dist_nm = dists.min(axis=1).astype(np.float32)
                    else:
                        lig_dist_nm = np.zeros((n_frames,), dtype=np.float32)
                except Exception:
                    lig_dist_nm = np.zeros((n_frames,), dtype=np.float32)
                glob_block = np.concatenate([glob_block, lig_dist_nm.reshape(-1, 1)], axis=1)

            # Append pocket descriptors if requested: pocket_CA_RMSD + N_contacts.
            # Must use the full-system trajectory (_tr_full_for_ligand) since `tr` is protein-only.
            if token_set in {"rmsd_e2e_pocket", "rmsd_e2e_pocket3", "rmsd_e2e_pocket_contacts", "rmsd_e2e_pocket_full"}:
                tr_pocket = _tr_full_for_ligand if _tr_full_for_ligand is not None else None
                # If no full-system traj available, try reloading with simulation_topology
                if tr_pocket is None:
                    sim_top_cand = base_dir_topology.get(bdir)
                    if sim_top_cand:
                        try:
                            tr_pocket = md.load(fpath, top=sim_top_cand, stride=max(1, stride))
                        except Exception:
                            tr_pocket = None
                if tr_pocket is None:
                    tr_pocket = tr
                try:
                    POCKET_CUTOFF_NM = 0.5
                    lig_names = {"UNK", "BEN", "LIG"}
                    lig_heavy_idx = np.array([a.index for a in tr_pocket.topology.atoms
                                              if a.residue.name in lig_names and a.element.symbol != "H"], dtype=int)
                    prot_heavy_idx = np.array([a.index for r in tr_pocket.topology.residues
                                               if r.name.upper() in _AA_SET
                                               for a in r.atoms if a.element.symbol != "H"], dtype=int)
                    # Detect pocket from the REFERENCE PDB (bound state), not from
                    # each trajectory's frame 0 (ligand may have dissociated during MD).
                    if pocket_ref_pdb and os.path.isfile(pocket_ref_pdb):
                        import mdtraj as _md_pref
                        _ref_for_pocket = _md_pref.load(pocket_ref_pdb)
                    else:
                        _ref_for_pocket = ref_full  # full PDB with ligand
                    _ref_lig_heavy = np.array([a.index for a in _ref_for_pocket.topology.atoms
                                               if a.residue.name in lig_names and a.element.symbol != "H"], dtype=int)
                    _ref_prot_res = [r for r in _ref_for_pocket.topology.residues if r.name.upper() in _AA_SET]
                    if len(_ref_lig_heavy) == 0:
                        raise ValueError("No ligand atoms in reference PDB")
                    _ref_lig_xyz = _ref_for_pocket.xyz[0, _ref_lig_heavy]
                    pocket_res_set = set()
                    for ri, r in enumerate(_ref_prot_res):
                        r_heavy = [a.index for a in r.atoms if a.element.symbol != "H"]
                        if not r_heavy: continue
                        d = np.min(np.linalg.norm(
                            _ref_for_pocket.xyz[0, r_heavy, :][:, None, :] - _ref_lig_xyz[None, :, :], axis=-1))
                        if d < POCKET_CUTOFF_NM:
                            pocket_res_set.add(ri)
                    # Map pocket residue indices from ref PDB to trajectory atom indices
                    traj_prot_res = [r for r in tr_pocket.topology.residues if r.name.upper() in _AA_SET]
                    pocket_ca_idx = np.array([a.index for ri in sorted(pocket_res_set)
                                              if ri < len(traj_prot_res)
                                              for a in traj_prot_res[ri].atoms if a.name == "CA"], dtype=int)
                    if len(pocket_ca_idx) < 2:
                        raise ValueError(f"Pocket too small ({len(pocket_ca_idx)} CA, {len(_ref_lig_heavy)} lig atoms in ref)")
                    # Reference pocket CA from the reference PDB (bound conformation)
                    ref_pocket_ca_from_pdb = np.array([a.index for ri in sorted(pocket_res_set)
                                                       if ri < len(_ref_prot_res)
                                                       for a in _ref_prot_res[ri].atoms if a.name == "CA"], dtype=int)
                    ref_pocket_ca = _ref_for_pocket.xyz[0, ref_pocket_ca_from_pdb].astype(np.float64)
                    # Per-frame: pocket CA RMSD
                    pocket_rmsd = np.zeros(n_frames, dtype=np.float32)
                    for fi in range(n_frames):
                        cur = tr_pocket.xyz[fi, pocket_ca_idx].astype(np.float64)
                        Pc = cur - cur.mean(0); Qc = ref_pocket_ca - ref_pocket_ca.mean(0)
                        H = Pc.T @ Qc; U, _, Vt = np.linalg.svd(H); V = Vt.T
                        d = np.sign(np.linalg.det(V @ U.T))
                        pocket_rmsd[fi] = np.sqrt(np.mean(np.sum(
                            (Pc @ (V @ np.diag([1.,1.,d]) @ U.T).T - Qc)**2, axis=1)))
                    # Per-frame: N contacts (protein-ligand heavy pairs within 4Å)
                    n_contacts = np.zeros(n_frames, dtype=np.float32)
                    for fi in range(n_frames):
                        dists = np.linalg.norm(
                            tr_pocket.xyz[fi, prot_heavy_idx, :][:, None, :] -
                            tr_pocket.xyz[fi, lig_heavy_idx, :][None, :, :], axis=-1)
                        n_contacts[fi] = float((dists < 0.4).sum())
                    n_contacts_scaled = n_contacts / 100.0
                    # Per-frame: lig-pocket COM distance (BEN heavy COM ↔ pocket Cα COM)
                    lig_pocket_dist = np.zeros(n_frames, dtype=np.float32)
                    if len(lig_heavy_idx) > 0 and len(pocket_ca_idx) > 0:
                        for fi in range(n_frames):
                            lig_com = tr_pocket.xyz[fi, lig_heavy_idx].mean(axis=0)
                            pocket_com = tr_pocket.xyz[fi, pocket_ca_idx].mean(axis=0)
                            lig_pocket_dist[fi] = float(np.linalg.norm(lig_com - pocket_com))
                    if token_set == "rmsd_e2e_pocket3":
                        glob_block = np.concatenate([glob_block,
                                                     pocket_rmsd.reshape(-1, 1)], axis=1)
                    elif token_set == "rmsd_e2e_pocket_full":
                        glob_block = np.concatenate([glob_block,
                                                     pocket_rmsd.reshape(-1, 1),
                                                     n_contacts_scaled.reshape(-1, 1),
                                                     lig_pocket_dist.reshape(-1, 1)], axis=1)
                    else:
                        glob_block = np.concatenate([glob_block,
                                                     pocket_rmsd.reshape(-1, 1),
                                                     n_contacts_scaled.reshape(-1, 1)], axis=1)
                    if first_traj_for_pocket_log:
                        print(f"[TorsionAR Pretrain] Pocket: {len(pocket_res_set)} residues, "
                              f"{len(pocket_ca_idx)} CA atoms (auto-detected within {POCKET_CUTOFF_NM*10:.0f}Å of ligand)",
                              flush=True)
                        first_traj_for_pocket_log = False
                except Exception as exc_pocket:
                    print(f"[TorsionAR Pretrain] Warning: pocket descriptor failed ({exc_pocket}); using zeros.", flush=True)
                    if token_set == "rmsd_e2e_pocket3":
                        n_pocket_cols = 1
                    elif token_set == "rmsd_e2e_pocket_full":
                        n_pocket_cols = 3
                    else:
                        n_pocket_cols = 2
                    glob_block = np.concatenate([glob_block,
                                                 np.zeros((n_frames, n_pocket_cols), dtype=np.float32)], axis=1)

        torsion_blocks.append(tors_block)
        mask_blocks.append(mask_block)
        globals_blocks.append(glob_block)
        total_frames += n_frames
        print(f"[TorsionAR Pretrain] Collected {n_frames} frames from {fpath}")

    if not torsion_blocks:
        raise RuntimeError("[TorsionAR Pretrain] No usable frames collected from trajectory files.")

    torsions_all = np.concatenate(torsion_blocks, axis=0)
    mask_all = np.concatenate(mask_blocks, axis=0)
    globals_all = np.concatenate(globals_blocks, axis=0) if globals_blocks else np.zeros((torsions_all.shape[0], 0), dtype=np.float32)
    N_total = torsions_all.shape[0]

    # Append ligand residue type (unknown/fallback) — one per rotatable bond position.
    if n_ligand_torsions > 0:
        residue_types_with_lig = np.append(residue_types_1d, [fallback_idx] * n_ligand_torsions)
    else:
        residue_types_with_lig = residue_types_1d
    residue_types = np.tile(residue_types_with_lig[None, :], (N_total, 1))

    print(f"[TorsionAR Pretrain] Final dataset: {torsions_all.shape[0]} frames, "
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
    pdb_b_path: Optional[str] = None,
    stride: int = 1,
    global_token_set: str = "rmsd_e2e",
    torsion_mode: str = "backbone",
    ligand_smiles: str = "",
    reweight_zenodo_dir: str = "",
    reweight_ref_pdb: str = "",
    reweight_labels_npz: str = "",
    reweight_alpha: float = 1.0,
    pocket_ref_pdb: str = "",
) -> None:
    """
    Pretrain the TorsionARPolicy on a dataset of torsion sequences.

    Args:
        source_path:  Path to .npz file with torsion data OR directory of trajectory files (.dcd/.xtc).
        out_path:     Where to save the pretrained weights (torch .pt).
    """
    # Canonicalize legacy spellings (e.g. backbone_chi -> backbone+sidechain) so
    # both the CLI and programmatic callers reach the tokenizer's accepted values.
    torsion_mode = normalize_torsion_mode(torsion_mode)
    source_dirs = _split_path_list(source_path)
    if source_dirs and all(os.path.isdir(p) for p in source_dirs):
        if not pdb_path:
            raise ValueError("pdb_path must be provided when pretraining from trajectory directories.")
        # Note: endpoint PDBs may be solute-only *or* fully solvated (explicit-solvent replay).
        # The torsion pretrain pipeline slices trajectories and reference PDBs to protein-like atoms,
        # so solvent/ions in endpoint files are not fatal. We keep validation best-effort and do not
        # fail pretraining when solvent is present.
        if pdb_b_path:
            try:
                from ..utils.endpoints import validate_endpoint_pdbs

                res = validate_endpoint_pdbs(pdb_path, pdb_b_path)
                print(
                    f"[TorsionAR Pretrain][Endpoints] A/B validated: n_solute_residues={res.n_solute_residues} n_solute_atoms={res.n_solute_atoms}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[TorsionAR Pretrain][Endpoints] Note: skipping strict A/B validation ({exc}).", flush=True)
        torsions_np, angle_mask_np, residue_types_np, global_values_np = _build_torsion_dataset_from_dcd(
            base_dir=source_dirs,
            pdb_path=pdb_path,
            pdb_b_path=pdb_b_path,
            stride=stride,
            global_token_set=global_token_set,
            torsion_mode=torsion_mode,
            ligand_smiles=ligand_smiles,
            pocket_ref_pdb=pocket_ref_pdb,
        )
    else:
        torsions_np, angle_mask_np, residue_types_np, global_values_np = _load_npz_dataset(source_path, torsion_mode=torsion_mode)
    N, L, _ = torsions_np.shape
    print(f"[TorsionAR Pretrain] Loaded {N} sequences of length {L} from {source_path}")

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
            token_set_compact = token_set.replace(" ", "")
            if token_set in {"adk_domain4", "adk_domain_only", "adk_domain_no_rmsd", "adk_no_rmsd"} and n_global_tokens == len(
                ADK_DOMAIN_TOKEN_NAMES
            ):
                token_names = list(ADK_DOMAIN_TOKEN_NAMES)
            elif token_set in {
                "adk_domain4_rmsd_ab",
                "adk_domain4_rmsd_a_b",
                "adk_domain4_rmsd_ab_nm",
                "adk_domain4_rmsd_a_b_nm",
            } and n_global_tokens == 2 + len(ADK_DOMAIN_TOKEN_NAMES):
                token_names = ["rmsd_a_nm", "rmsd_b_nm", *list(ADK_DOMAIN_TOKEN_NAMES)]
            elif token_set in {"adk", "adk_domain", "adk_globals", "adk_thetas"} and n_global_tokens == len(ADK_GLOBAL_TOKEN_NAMES):
                token_names = list(ADK_GLOBAL_TOKEN_NAMES)
            elif token_set in {
                "rmsd_ab",
                "rmsd_a_b",
                "rmsd_ab_nm",
                "rmsd_a_b_nm",
            } and n_global_tokens == 2:
                token_names = ["rmsd_a_nm", "rmsd_b_nm"]
            elif token_set in {
                "rmsd_ab_e2e",
                "rmsd_a_b_e2e",
                "rmsd_ab_e2e_nm",
                "rmsd_a_b_e2e_nm",
                "path_rmsd_ab",
                "path_rmsd_ab_e2e",
            } and n_global_tokens == 3:
                token_names = ["rmsd_a_nm", "rmsd_b_nm", "e2e_nm"]
            elif token_set in {"vinculin_h5", "vinculin_h5_bundle", "h5_bundle"} and n_global_tokens == 5:
                from gespo.utils.vinculin_h5_metrics import VINCULIN_H5_TOKEN_NAMES
                token_names = list(VINCULIN_H5_TOKEN_NAMES)
            elif token_set_compact in {
                "ppi",
                "ppi_global",
                "ppi_globals",
                "ppi_tokens",
                "chain_com_dist_nm,min_interface_ca_dist_nm,q_interface",
            } and n_global_tokens == 3:
                from gespo.utils.ppi_metrics import PPI_GLOBAL_TOKEN_NAMES
                token_names = list(PPI_GLOBAL_TOKEN_NAMES)
            elif token_set in {"rmsd_e2e_pocket", "rmsd_e2e_pocket_contacts"} and n_global_tokens == 4:
                token_names = ["rmsd_nm", "e2e_nm", "pocket_rmsd_nm", "n_contacts_scaled"]
            elif token_set == "rmsd_e2e_pocket_full" and n_global_tokens == 5:
                token_names = ["rmsd_nm", "e2e_nm", "pocket_rmsd_nm", "n_contacts_scaled", "lig_pocket_dist_nm"]
            elif token_set in {"rmsd_e2e_pocket", "rmsd_e2e_pocket3"} and n_global_tokens == 3:
                token_names = ["rmsd_nm", "e2e_nm", "pocket_rmsd_nm"]
            elif n_global_tokens == 2:
                token_names = ["rmsd_nm", "e2e_nm"]
            elif n_global_tokens == 3:
                token_names = ["rmsd_nm", "e2e_nm", "rg_nm"]
            else:
                token_names = [f"global_{i}" for i in range(n_global_tokens)]
        else:
            dataset = torch.utils.data.TensorDataset(torsions, angle_mask, residue_types)
            n_global_tokens = 0
    else:
        dataset = torch.utils.data.TensorDataset(torsions, angle_mask, residue_types)
        n_global_tokens = 0

    # Quick global-token sanity check to catch unit/scale issues early (e.g., ADK dist/theta near 0).
    try:
        if n_global_tokens > 0 and global_values_np is not None and token_names is not None:
            g_np = np.asarray(global_values_np, dtype=np.float32)
            if g_np.ndim == 2 and g_np.shape[1] == n_global_tokens:
                print("[TorsionAR Pretrain] Global-token dataset stats (raw units):", flush=True)
                for j, name in enumerate(list(token_names)[:n_global_tokens]):
                    v = g_np[:, j]
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        continue
                    print(
                        f"  {name}: mean={float(v.mean()):.3f} std={float(v.std()):.3f} "
                        f"min={float(v.min()):.3f} max={float(v.max()):.3f}",
                        flush=True,
                    )
    except Exception:
        pass
    # ── Optional: importance reweighting to uniform-state distribution ──
    # If reweight_zenodo_dir is set, cluster training frames into 7 macrostates
    # using 3 loop RMSDs (Asp189/Trp215/Loop225) vs 3PTB, anchored on Zenodo
    # (Plattner & Noe 150μs) cluster centroids. Each frame gets weight
    # 1/(N_in_its_cluster) so each state contributes ~equal mass during training.
    sample_weights = None
    # ── Precomputed labels (clean path, no Plattner) ──
    if reweight_labels_npz:
        try:
            print(f"[TorsionAR Pretrain] Loading reweight labels from {reweight_labels_npz}", flush=True)
            data = np.load(reweight_labels_npz, allow_pickle=True)
            labels = np.asarray(data["labels"], dtype=np.int64)
            if labels.shape[0] != torsions.shape[0]:
                raise ValueError(f"labels.shape[0]={labels.shape[0]} != torsions.shape[0]={torsions.shape[0]}")
            n_clusters = int(labels.max()) + 1
            counts = np.bincount(labels, minlength=n_clusters)
            print(f"[TorsionAR Pretrain] Cluster counts: " +
                  " ".join(f"C{i}={int(counts[i])}({100*counts[i]/counts.sum():.1f}%)" for i in range(n_clusters)),
                  flush=True)
            alpha = float(reweight_alpha)
            w = np.empty(labels.shape[0], dtype=np.float32)
            for c in range(n_clusters):
                n_c = max(int(counts[c]), 1)
                w[labels == c] = 1.0 / float(n_c) ** alpha
            w = w / w.sum() * w.shape[0]
            sample_weights = torch.from_numpy(w).double()
            ess = float((sample_weights.sum() ** 2 / (sample_weights ** 2).sum()).item())
            print(f"[TorsionAR Pretrain] VAMP-cluster reweighting (alpha={alpha}): ESS = {ess:.1f} / {sample_weights.shape[0]}", flush=True)
        except Exception as exc_rl:
            print(f"[TorsionAR Pretrain] Warning: labels-npz reweighting failed ({exc_rl}); falling back to uniform.", flush=True)
            sample_weights = None

    if sample_weights is None and reweight_zenodo_dir:
        import glob as _glob
        from sklearn.cluster import KMeans
        _AA_SET = {"ALA","ARG","ASN","ASP","CYS","CYX","GLN","GLU","GLY","HIS","HID","HIE","HIP",
                   "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","ACE","NME"}
        try:
            print(f"[TorsionAR Pretrain] Computing importance weights using Zenodo at {reweight_zenodo_dir}", flush=True)
            zen_top_pdb = os.path.join(reweight_zenodo_dir, "nowat_noCa.pdb")
            zen_files = sorted(_glob.glob(os.path.join(reweight_zenodo_dir, "cnew-2micro-*.xtc")))
            ref_3ptb_path = reweight_ref_pdb or pdb_path

            YELLOW_LOOP_IDX = list(range(168, 176))
            GREEN_LOOP_IDX = list(range(192, 199))
            ORANGE_LOOP_IDX = list(range(202, 208))

            def _filter_aa(traj):
                aa_atoms = [a.index for r in traj.topology.residues
                            if r.name.upper() in _AA_SET for a in r.atoms]
                if aa_atoms and len(aa_atoms) < traj.n_atoms:
                    return traj.atom_slice(aa_atoms)
                return traj

            def _get_ca(t, idxs):
                prot = [r for r in t.topology.residues if r.name.upper() in _AA_SET]
                out = []
                for ri in idxs:
                    if ri >= len(prot):
                        continue
                    for a in prot[ri].atoms:
                        if a.name == "CA":
                            out.append(a.index); break
                return np.array(out, dtype=np.int64)

            def _kabsch_rmsd(P, Q):
                Pc = P - P.mean(0); Qc = Q - Q.mean(0)
                H = Pc.T @ Qc
                U, _, Vt = np.linalg.svd(H)
                V = Vt.T
                d = np.sign(np.linalg.det(V @ U.T))
                D = np.diag([1.0, 1.0, d])
                R = V @ D @ U.T
                return np.sqrt(np.mean(np.sum((Pc @ R.T - Qc) ** 2, axis=1)))

            # 3PTB reference (filtered to AA)
            ref_traj = _filter_aa(md.load(ref_3ptb_path))
            ref_y = ref_traj.xyz[0, _get_ca(ref_traj, YELLOW_LOOP_IDX), :].astype(np.float64)
            ref_g = ref_traj.xyz[0, _get_ca(ref_traj, GREEN_LOOP_IDX), :].astype(np.float64)
            ref_o = ref_traj.xyz[0, _get_ca(ref_traj, ORANGE_LOOP_IDX), :].astype(np.float64)

            # Fit clusters on Zenodo (load XTC with FULL topology, then filter)
            zen_top_full = md.load(zen_top_pdb)
            zen_feats = []
            # Get AA-filtered version (used for CA selection)
            zen_top_aa = _filter_aa(zen_top_full)
            z_y_ca = _get_ca(zen_top_aa, YELLOW_LOOP_IDX)
            z_g_ca = _get_ca(zen_top_aa, GREEN_LOOP_IDX)
            z_o_ca = _get_ca(zen_top_aa, ORANGE_LOOP_IDX)
            for f in zen_files:
                tr_z = md.load(f, top=zen_top_full.topology, stride=20)
                tr_z = _filter_aa(tr_z)
                for i in range(tr_z.n_frames):
                    y = 10 * _kabsch_rmsd(tr_z.xyz[i, z_y_ca, :].astype(np.float64), ref_y)
                    g = 10 * _kabsch_rmsd(tr_z.xyz[i, z_g_ca, :].astype(np.float64), ref_g)
                    o = 10 * _kabsch_rmsd(tr_z.xyz[i, z_o_ca, :].astype(np.float64), ref_o)
                    zen_feats.append([y, g, o])
            zen_feats = np.asarray(zen_feats, dtype=np.float32)
            km = KMeans(n_clusters=7, n_init=30, random_state=42)
            km.fit(zen_feats)
            print(f"[TorsionAR Pretrain] Zenodo cluster centroids (Y, G, O in Å):", flush=True)
            for s in range(7):
                c = km.cluster_centers_[s]
                print(f"  S{s}: Y={c[0]:.2f} G={c[1]:.2f} O={c[2]:.2f}", flush=True)

            # Compute training-data loop RMSDs (re-load trajectories)
            train_feats = []
            for bdir in source_dirs:
                if not os.path.isdir(bdir):
                    continue
                for ext in ("*.dcd", "*.xtc"):
                    for fpath in sorted(_glob.glob(os.path.join(bdir, "**", ext), recursive=True)):
                        try:
                            local_top_path = None
                            for cand_pdb in [os.path.join(os.path.dirname(fpath), "simulation_topology.pdb"),
                                              os.path.join(os.path.dirname(os.path.dirname(fpath)), "simulation_topology.pdb")]:
                                if os.path.exists(cand_pdb):
                                    local_top_path = cand_pdb; break
                            local_top = md.load(local_top_path).topology if local_top_path else md.load(ref_3ptb_path).topology
                            tr_t = md.load(fpath, top=local_top, stride=max(1, stride))
                            tr_t = _filter_aa(tr_t)
                            t_y_ca = _get_ca(tr_t, YELLOW_LOOP_IDX)
                            t_g_ca = _get_ca(tr_t, GREEN_LOOP_IDX)
                            t_o_ca = _get_ca(tr_t, ORANGE_LOOP_IDX)
                            for i in range(tr_t.n_frames):
                                y = 10 * _kabsch_rmsd(tr_t.xyz[i, t_y_ca, :].astype(np.float64), ref_y)
                                g = 10 * _kabsch_rmsd(tr_t.xyz[i, t_g_ca, :].astype(np.float64), ref_g)
                                o = 10 * _kabsch_rmsd(tr_t.xyz[i, t_o_ca, :].astype(np.float64), ref_o)
                                train_feats.append([y, g, o])
                        except Exception as exc_t:
                            print(f"[TorsionAR Pretrain] Warning: reweight feature extract failed for {fpath}: {exc_t}", flush=True)
                            continue
            train_feats = np.asarray(train_feats, dtype=np.float32)
            if train_feats.shape[0] != torsions.shape[0]:
                print(f"[TorsionAR Pretrain] Warning: train_feats {train_feats.shape[0]} != torsions {torsions.shape[0]}; reweighting disabled.", flush=True)
            else:
                train_labels = km.predict(train_feats)
                state_counts = np.bincount(train_labels, minlength=7)
                print(f"[TorsionAR Pretrain] Train state populations: " +
                      " ".join(f"S{i}={c}({100*c/state_counts.sum():.1f}%)" for i, c in enumerate(state_counts)),
                      flush=True)
                # Weight: 1/N_in_state → uniform state mass during sampling.
                w = np.empty(train_labels.shape[0], dtype=np.float32)
                for s in range(7):
                    n_s = max(int(state_counts[s]), 1)
                    w[train_labels == s] = 1.0 / float(n_s)
                w = w / w.sum() * w.shape[0]
                sample_weights = torch.from_numpy(w).double()
                print(f"[TorsionAR Pretrain] Computed importance weights; effective sample size = "
                      f"{(sample_weights.sum() ** 2 / (sample_weights ** 2).sum()).item():.1f} / {sample_weights.shape[0]}",
                      flush=True)
        except Exception as exc_rw:
            print(f"[TorsionAR Pretrain] Warning: reweighting failed ({exc_rw}); using uniform sampling.", flush=True)
            sample_weights = None

    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=False,
        )
        print(f"[TorsionAR Pretrain] Using WeightedRandomSampler for uniform-state training.", flush=True)
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        )

    model = TorsionARPolicy(
        num_residues=L,
        residue_vocab_size=residue_vocab,
        d_model=d_model,
        nhead=n_heads,
        num_layers=n_layers,
        dropout=0.1,
        kmix=kmix,
        n_angles=int(torsions.shape[2]),
        n_global_tokens=n_global_tokens,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[TorsionAR Pretrain] Model initialized on device {device}")

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
        print(f"[TorsionAR Pretrain] Epoch {ep+1}/{epochs} - NLL per sequence: {avg_loss:.4f}")

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
            "n_angles": int(model.n_angles),
            "angle_names": list(getattr(model, "angle_names", [])) if getattr(model, "angle_names", None) is not None else None,
            "torsion_mode": ("backbone+sidechain" if int(getattr(model, "n_angles", 3) or 3) > 3 else "backbone"),
            "n_global_tokens": n_global_tokens,
            "global_token_names": list(token_names) if token_names is not None else None,
            "global_token_set": str(global_token_set or "").strip().lower() if n_global_tokens > 0 else "",
        },
    }
    torch.save(payload, out_path)
    print(f"[TorsionAR Pretrain] Saved pretrained TorsionARPolicy (with config) to {out_path}")
