"""Geometry utilities: residue maps, Kabsch alignment, dihedrals and conversions."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mdtraj as md
import numpy as np
import torch
from openmm.app import Topology

def find_all_backbone_dihedrals(topology: Topology) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Build backbone (phi, psi) dihedral definitions.

    Notes:
    - We define dihedrals for *protein residues* (standard amino acids) but allow
      the previous/next residue to be a cap (ACE/NME) when present.
    - The previous implementation relied on residue.id strings like "ACE"/"NME",
      which breaks on typical PDBs where residue IDs are numeric (e.g., ACE=1,
      ALA=2, NME=3). This implementation uses chain order instead.
    """

    dihedrals: List[Tuple[str, Tuple[int, int, int, int]]] = []

    # Exclude water, caps, and ions from the *target* residue set (but caps are
    # still eligible as neighbors when constructing phi/psi).
    NON_PROTEIN = {
        "HOH",
        "WAT",
        "SOL",
        "TIP3",
        "TIP3P",
        "ACE",
        "NME",
        "NA",
        "CL",
        "K",
        "CA",
        "MG",
        "ZN",
        "FE",
        "CU",
        "SOD",
        "CLA",
        "POP",
        "Na+",
        "Cl-",
        "K+",
        "Ca2+",
        "Mg2+",
        "Zn2+",
    }

    atom_map = {(atom.residue.index, atom.name): atom.index for atom in topology.atoms()}

    def _atom_idx(res_idx: int, atom_name: str) -> Optional[int]:
        try:
            return int(atom_map[(int(res_idx), str(atom_name))])
        except Exception:
            return None

    # Iterate per-chain to respect residue order and avoid cross-chain dihedrals.
    for chain in topology.chains():
        chain_residues = list(chain.residues())
        for local_i, residue in enumerate(chain_residues):
            # Only define torsions for protein residues.
            if str(residue.name) in NON_PROTEIN:
                continue

            res_idx = int(residue.index)

            # Backbone atoms on the current residue.
            n = _atom_idx(res_idx, "N")
            ca = _atom_idx(res_idx, "CA")
            c = _atom_idx(res_idx, "C")
            if n is None or ca is None or c is None:
                continue

            # phi: C(prev) - N - CA - C
            if local_i - 1 >= 0:
                prev = chain_residues[local_i - 1]
                c_prev = _atom_idx(int(prev.index), "C")
                if c_prev is not None:
                    dihedrals.append((f"phi_{res_idx + 1}", (c_prev, n, ca, c)))

            # psi: N - CA - C - N(next)
            if local_i + 1 < len(chain_residues):
                nxt = chain_residues[local_i + 1]
                n_next = _atom_idx(int(nxt.index), "N")
                if n_next is not None:
                    dihedrals.append((f"psi_{res_idx + 1}", (n, ca, c, n_next)))

    dihedrals.sort(key=lambda x: (int(x[0].split("_")[1]), x[0].split("_")[0]))
    return dihedrals


def find_ligand_rotatable_dihedrals_smiles(
    md_topology: md.Topology,
    ligand_smiles: str,
    ligand_resnames: Optional[List[str]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """RDKit + SMILES + standard SMARTS rotatable bond detection.

    Uses the canonical SMARTS pattern `[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]`
    after assigning bond orders from a SMILES template. Allows
    hydrogen as i/l atom when no heavy neighbor is available
    (e.g., amidine C-N rotation where N has only H attached).

    Returns list of (label, (i, j, k, l)) using global atom indices.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return []

    template = Chem.MolFromSmiles(str(ligand_smiles))
    if template is None:
        return []
    template = Chem.AddHs(template)
    rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

    PROTEIN_RESNAMES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
        "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NME",
        "HOH", "WAT", "TIP3", "NA", "CL", "K", "CA2",
    }

    dihedrals: List[Tuple[str, Tuple[int, int, int, int]]] = []

    for res in md_topology.residues:
        if ligand_resnames and res.name not in ligand_resnames:
            continue
        if not ligand_resnames and res.name.upper() in PROTEIN_RESNAMES:
            continue

        atoms = list(res.atoms)
        if len(atoms) < 4:
            continue

        # Build a minimal RDKit Mol from topology atoms + intra-residue bonds.
        emol = Chem.RWMol()
        local_to_global: List[int] = []
        for a in atoms:
            emol.AddAtom(Chem.Atom(a.element.symbol))
            local_to_global.append(a.index)
        global_to_local = {gi: li for li, gi in enumerate(local_to_global)}

        for b in md_topology.bonds:
            a1, a2 = b[0], b[1]
            if a1.residue.index != res.index or a2.residue.index != res.index:
                continue
            li1 = global_to_local.get(a1.index)
            li2 = global_to_local.get(a2.index)
            if li1 is not None and li2 is not None:
                emol.AddBond(li1, li2, Chem.BondType.SINGLE)

        topology_mol = emol.GetMol()

        # Assign correct bond orders by matching the SMILES template.
        try:
            topology_mol = AllChem.AssignBondOrdersFromTemplate(template, topology_mol)
        except Exception:
            continue

        # Find rotatable bonds.
        matches = topology_mol.GetSubstructMatches(rot_smarts)
        seen_bonds: set = set()
        for m in matches:
            a_local, b_local = sorted([int(m[0]), int(m[1])])
            if (a_local, b_local) in seen_bonds:
                continue
            seen_bonds.add((a_local, b_local))

            atom_a = topology_mol.GetAtomWithIdx(a_local)
            atom_b = topology_mol.GetAtomWithIdx(b_local)

            def _pick_neighbor(atom, exclude_idx):
                heavy = None
                any_atom = None
                for n in atom.GetNeighbors():
                    if n.GetIdx() == exclude_idx:
                        continue
                    if any_atom is None:
                        any_atom = n.GetIdx()
                    if n.GetSymbol() != "H":
                        heavy = n.GetIdx()
                        break
                return heavy if heavy is not None else any_atom

            i_local = _pick_neighbor(atom_a, b_local)
            l_local = _pick_neighbor(atom_b, a_local)
            if i_local is None or l_local is None:
                continue

            label = f"{res.name}{res.resSeq}_rot_{atoms[a_local].name}-{atoms[b_local].name}"
            dihedrals.append((label, (
                int(local_to_global[i_local]),
                int(local_to_global[a_local]),
                int(local_to_global[b_local]),
                int(local_to_global[l_local]),
            )))

    return dihedrals


def find_ligand_rotatable_dihedrals(
    md_topology: md.Topology,
    ligand_resnames: Optional[List[str]] = None,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Legacy heavy-atoms-only rotatable bond detection.

    Iterates residue bonds and selects non-ring, non-terminal heavy-heavy bonds.
    Misses rotatable bonds where one side has only H neighbors (e.g., amidine C-N).
    Prefer find_ligand_rotatable_dihedrals_smiles() when a SMILES is available.

    Returns list of (label, (i, j, k, l)) using global atom indices.
    """
    PROTEIN_RESNAMES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
        "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE",
        "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NME",
        "HOH", "WAT", "TIP3", "NA", "CL", "K", "CA2",
    }

    dihedrals: List[Tuple[str, Tuple[int, int, int, int]]] = []

    for res in md_topology.residues:
        if ligand_resnames and res.name not in ligand_resnames:
            continue
        if not ligand_resnames and res.name.upper() in PROTEIN_RESNAMES:
            continue

        atoms = list(res.atoms)
        heavy = [a for a in atoms if a.element.symbol != "H"]
        if len(heavy) < 4:
            continue

        heavy_idx = {a.index for a in heavy}
        bonds = []
        for b in md_topology.bonds:
            a1, a2 = b[0], b[1]
            if a1.residue.index == res.index and a2.residue.index == res.index:
                if a1.index in heavy_idx and a2.index in heavy_idx:
                    bonds.append((a1, a2))

        for a1, a2 in bonds:
            if a1.element.symbol == "H" or a2.element.symbol == "H":
                continue
            in_ring = _atoms_in_same_ring(a1, a2, md_topology, res)
            if in_ring:
                continue
            n1 = [x for x in _bonded_heavy(a1, md_topology, res) if x.index != a2.index]
            n2 = [x for x in _bonded_heavy(a2, md_topology, res) if x.index != a1.index]
            if not n1 or not n2:
                continue
            i_atom = max(n1, key=lambda a: a.element.mass)
            l_atom = max(n2, key=lambda a: a.element.mass)
            label = f"{res.name}{res.resSeq}_rot_{a1.name}-{a2.name}"
            dihedrals.append((label, (i_atom.index, a1.index, a2.index, l_atom.index)))

    return dihedrals


def _bonded_heavy(atom, topology, residue):
    """Return heavy atoms bonded to `atom` within the same residue."""
    result = []
    for b in topology.bonds:
        a1, a2 = b[0], b[1]
        if a1.index == atom.index and a2.residue.index == residue.index and a2.element.symbol != "H":
            result.append(a2)
        elif a2.index == atom.index and a1.residue.index == residue.index and a1.element.symbol != "H":
            result.append(a1)
    return result


def _atoms_in_same_ring(a1, a2, topology, residue):
    """Check if two atoms are in the same ring (simple BFS-based)."""
    adj = {}
    for b in topology.bonds:
        b1, b2 = b[0], b[1]
        if b1.residue.index == residue.index and b2.residue.index == residue.index:
            if b1.element.symbol != "H" and b2.element.symbol != "H":
                adj.setdefault(b1.index, []).append(b2.index)
                adj.setdefault(b2.index, []).append(b1.index)
    visited = {a1.index}
    queue = [(n, [a1.index, n]) for n in adj.get(a1.index, []) if n != a2.index]
    while queue:
        node, path = queue.pop(0)
        if node == a2.index and len(path) >= 3:
            return True
        if node in visited:
            continue
        visited.add(node)
        for nb in adj.get(node, []):
            if nb not in visited or (nb == a2.index and len(path) >= 2):
                queue.append((nb, path + [nb]))
    return False


def find_all_sidechain_chi_dihedrals(
    topology: Topology,
    *,
    max_chi: int = 4,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Build sidechain χ dihedral definitions (χ1..χK) using MDTraj's residue templates.

    Notes:
    - Atom indices are returned in the OpenMM Topology/System index space.
    - Names follow the same convention as backbone dihedrals: '{chiK}_{res_idx+1}'.
    - Not all residues have the same number of χ angles; missing angles are simply omitted.
    """
    max_chi_i = int(max_chi)
    if max_chi_i <= 0:
        return []

    md_top = md.Topology.from_openmm(topology)
    if int(md_top.n_atoms) == 0:
        return []

    # MDTraj dihedral helpers require a Trajectory (for coordinates), but the torsion
    # *definitions* depend only on topology. Use a small random cloud to avoid
    # degenerate collinearities that can cause numerical issues.
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(1, int(md_top.n_atoms), 3)).astype(np.float32)
    tr = md.Trajectory(xyz, md_top)

    dihedrals: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for k in range(1, max_chi_i + 1):
        fn = getattr(md, f"compute_chi{k}", None)
        if fn is None:
            continue
        try:
            idx, _ = fn(tr)
        except Exception:
            continue
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 4)
        for tors in idx:
            try:
                res_idx = int(md_top.atom(int(tors[1])).residue.index)
            except Exception:
                continue
            dihedrals.append(
                (
                    f"chi{k}_{res_idx + 1}",
                    (int(tors[0]), int(tors[1]), int(tors[2]), int(tors[3])),
                )
            )

    def _sort_key(item: Tuple[str, Tuple[int, int, int, int]]):
        name = item[0]
        try:
            kind, rid = name.split("_", 1)
            chi_k = int(kind[3:]) if kind.startswith("chi") else 0
            rid_i = int(rid)
        except Exception:
            chi_k = 0
            rid_i = 0
        return (rid_i, chi_k)

    dihedrals.sort(key=_sort_key)
    return dihedrals

def _dihedral(pos: np.ndarray, atoms: Tuple[int, int, int, int]) -> float:
    r = [pos[n] for n in atoms]
    b0, b1, b2 = r[0] - r[1], r[2] - r[1], r[3] - r[2]
    b1 /= np.linalg.norm(b1) + 1e-6
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x, y = np.dot(v, w), np.dot(np.cross(b1, v), w)
    return math.atan2(y, x)

def build_residue_maps(
    traj: md.Trajectory,
    *,
    atom_indices: Optional[np.ndarray] = None,
    residue_filter=None,
):
    """Build residue-group padding/masks for hierarchical models.

    Parameters
    ----------
    traj:
        MDTraj trajectory whose topology defines residue/atom ordering.
    atom_indices:
        Optional atom indices (into ``traj``) to restrict the mapping (e.g., protein-only).
        When provided, only atoms in this set are considered.
    residue_filter:
        Optional predicate ``(residue) -> bool``. When provided, only residues for which
        this returns True are included.

    Returns
    -------
    res_idx_pad, res_mask, heavy_atom_indices
        ``res_idx_pad``/``res_mask`` index into the heavy-atom coordinate array built by
        selecting ``heavy_atom_indices`` from the original coordinates.
    """

    selected_set = None
    if atom_indices is not None:
        atom_indices = np.asarray(atom_indices, dtype=np.int64).reshape(-1)
        selected_set = set(int(i) for i in atom_indices if int(i) >= 0)

    def _keep_atom(atom) -> bool:
        if selected_set is not None and int(atom.index) not in selected_set:
            return False
        if residue_filter is not None:
            try:
                if not bool(residue_filter(atom.residue)):
                    return False
            except Exception:
                return False
        return True

    heavy_atom_indices = [
        int(atom.index)
        for atom in traj.topology.atoms
        if _keep_atom(atom) and atom.element.symbol != "H"
    ]
    global_to_light = {g: i for i, g in enumerate(heavy_atom_indices)}
    residue_groups = []
    for residue in traj.topology.residues:
        if residue_filter is not None:
            try:
                if not bool(residue_filter(residue)):
                    continue
            except Exception:
                continue
        heavy = [
            global_to_light[int(atom.index)]
            for atom in residue.atoms
            if _keep_atom(atom) and atom.element.symbol != "H" and int(atom.index) in global_to_light
        ]
        if heavy:
            residue_groups.append(heavy)
    R = len(residue_groups)
    if R == 0:
        raise ValueError("No residues selected; cannot build residue maps.")
    max_atoms_res = max(len(g) for g in residue_groups)
    res_idx_pad = torch.zeros((R, max_atoms_res), dtype=torch.long)
    res_mask = torch.zeros((R, max_atoms_res), dtype=torch.bool)
    for r, atom_idxs in enumerate(residue_groups):
        res_idx_pad[r, : len(atom_idxs)] = torch.tensor(atom_idxs, dtype=torch.long)
        res_mask[r, : len(atom_idxs)] = True
    return res_idx_pad, res_mask, heavy_atom_indices

def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_center = pred.mean(axis=0)
    target_center = target.mean(axis=0)
    pred_zero = pred - pred_center
    target_zero = target - target_center
    H = pred_zero.T @ target_zero
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = pred_zero @ R + target_center
    return aligned

def round_and_wrap_angles(angles: np.ndarray, decimals: int) -> np.ndarray:
    if decimals >= 0:
        angles = np.round(angles, decimals=decimals)
    angles = ((angles + np.pi) % (2 * np.pi)) - np.pi
    return angles.astype(np.float32)

def cv_targets_to_phi_psi(cv_targets: np.ndarray, cv_definitions: List[Tuple[str, Tuple[int, int, int, int]]]) -> Tuple[np.ndarray, np.ndarray]:
    phi_pairs, psi_pairs = [], []
    for idx, (name, _) in enumerate(cv_definitions):
        # Safety check: skip if index is out of bounds (cv_definitions may include pairwise/global CVs)
        if idx >= len(cv_targets):
            break
        if name.startswith('phi_'):
            res_id = int(name.split('_')[1])
            phi_pairs.append((res_id, cv_targets[idx]))
        elif name.startswith('psi_'):
            res_id = int(name.split('_')[1])
            psi_pairs.append((res_id, cv_targets[idx]))
    phi_pairs.sort(key=lambda x: x[0])
    psi_pairs.sort(key=lambda x: x[0])
    phi_angles = np.array([angle for _, angle in phi_pairs], dtype=np.float32)
    psi_angles = np.array([angle for _, angle in psi_pairs], dtype=np.float32)
    return phi_angles, psi_angles

def calculate_rmsd(coords_traj: np.ndarray, ref_coords: np.ndarray, atom_indices=None) -> np.ndarray:
    """
    Calculates the RMSD of a trajectory with respect to a reference structure,
    after aligning each frame to the reference using the Kabsch algorithm.
    Can operate on a subset of atoms (e.g., C-alphas).
    Vectorized for efficiency.
    """
    # Convert ref_coords to numpy array if it's not already
    if not isinstance(ref_coords, np.ndarray):
        ref_coords = np.array(ref_coords)
    
    if isinstance(atom_indices, list):
        atom_indices = np.array(atom_indices).astype(int)

    if atom_indices is None:
        atom_indices = slice(None)  # Use all atoms
    else:
        pass  # Silence verbose RMSD selection prints
    
    # Subset coordinates
    ref_subset = ref_coords[atom_indices]  # (n_atoms, 3)
    traj_subset = coords_traj[:, atom_indices]  # (n_frames, n_atoms, 3)

    # Prepare reference
    ref_centroid = ref_subset.mean(axis=0)
    ref_centered = ref_subset - ref_centroid

    n_frames = traj_subset.shape[0]
    n_atoms = ref_subset.shape[0]

    # Chunking to cap peak memory
    bytes_per_frame = n_atoms * 3 * 8  # float64
    target_bytes = 100 * 1024 * 1024  # 100 MB
    chunk_size = max(1, int(target_bytes // max(bytes_per_frame, 1)))
    if chunk_size > n_frames:
        chunk_size = n_frames

    rmsd_out = np.empty(n_frames, dtype=np.float64)
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        chunk = traj_subset[start:end]
        traj_centroids = chunk.mean(axis=1, keepdims=True)
        traj_centered = chunk - traj_centroids
        H = np.einsum('fji,jk->fik', traj_centered, ref_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.einsum('fij,fjk->fik', U, Vt)
        det_R = np.linalg.det(R)
        mask = det_R < 0
        if np.any(mask):
            Uc = U.copy()
            Uc[mask, :, -1] *= -1
            R[mask] = np.einsum('fij,fjk->fik', Uc[mask], Vt[mask])
        rotated = np.einsum('fij,fjk->fik', traj_centered, R)
        diff = rotated - ref_centered[np.newaxis, :, :]
        squared = np.sum(diff**2, axis=2)
        rmsd_out[start:end] = np.sqrt(np.mean(squared, axis=1))
        start = end

    return rmsd_out
