"""Geometry utilities: residue maps, Kabsch alignment, dihedrals and conversions."""
from __future__ import annotations

from .._imports import *

def find_all_backbone_dihedrals(topology: Topology) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    dihedrals = []
    residues = [r for r in topology.residues() if r.name not in ['HOH', 'ACE', 'NME']]
    atom_map = {(atom.residue.id, atom.name): atom for atom in topology.atoms()}
    
    for i in range(len(residues)):
        try:
            res_prev_id = residues[i-1].id if i > 0 else 'ACE'
            res_curr_id = residues[i].id
            atoms = (atom_map[(res_prev_id, 'C')].index, atom_map[(res_curr_id, 'N')].index,
                     atom_map[(res_curr_id, 'CA')].index, atom_map[(res_curr_id, 'C')].index)
            dihedrals.append((f"phi_{residues[i].index+1}", atoms))
        except KeyError: continue
        
    for i in range(len(residues)):
        try:
            res_curr_id = residues[i].id
            res_next_id = residues[i+1].id if i < len(residues) - 1 else 'NME'
            atoms = (atom_map[(res_curr_id, 'N')].index, atom_map[(res_curr_id, 'CA')].index,
                     atom_map[(res_curr_id, 'C')].index, atom_map[(res_next_id, 'N')].index)
            dihedrals.append((f"psi_{residues[i].index+1}", atoms))
        except KeyError: continue
    
    dihedrals.sort(key=lambda x: (int(x[0].split('_')[1]), x[0].split('_')[0]))
    return dihedrals

def _dihedral(pos: np.ndarray, atoms: Tuple[int, int, int, int]) -> float:
    r = [pos[n] for n in atoms]
    b0, b1, b2 = r[0] - r[1], r[2] - r[1], r[3] - r[2]
    b1 /= np.linalg.norm(b1) + 1e-6
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x, y = np.dot(v, w), np.dot(np.cross(b1, v), w)
    return math.atan2(y, x)

def build_residue_maps(traj: md.Trajectory):
    heavy_atom_indices = [atom.index for atom in traj.topology.atoms if atom.element.symbol != 'H']
    global_to_light = {g: i for i, g in enumerate(heavy_atom_indices)}
    residue_groups = []
    for residue in traj.topology.residues:
        heavy = [global_to_light[atom.index] for atom in residue.atoms if atom.element.symbol != 'H']
        if heavy:
            residue_groups.append(heavy)
    R = len(residue_groups)
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
