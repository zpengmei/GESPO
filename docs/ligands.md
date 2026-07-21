# Protein–ligand runs

GESPO handles a bound ligand end to end from a SMILES string — no external ligand
parameterization files are needed. Reference recipe:
[`../examples/protein_ligand_trypsin/`](../examples/protein_ligand_trypsin/).

## Ligand parameterization (SMILES → GAFF)

Pass `--ligand-smiles "<SMILES>"` to `seed`, `pretrain`, and `train`. At runtime GESPO:

1. builds the molecule from SMILES (OpenFF toolkit),
2. registers a **GAFF-2.11** template (`openmmforcefields.GAFFTemplateGenerator`), and
3. lets OpenMM parameterize the ligand on the fly.

The ligand must exist as a residue in your `--pdb` (commonly `BEN`, `LIG`, or `UNK`).
The protein/water use the usual `--forcefield-protein` / `--forcefield-water`.
Example: benzamidine is `c1ccc(cc1)C(=N)N` (2 rotatable dihedrals; RDKit detects them).

## Ligand torsion tokens

With `--ligand-smiles`, `gespo pretrain` adds the ligand's rotatable dihedrals as
extra positions in the TorsionAR sequence, so the policy models ligand internal
motion alongside the protein backbone. Pair it with a pocket-aware token set:

- **holo `--pdb`** (ligand present): `--global-token-set rmsd_e2e_pocket_full`
  (RMSD, e2e, pocket RMSD, protein–ligand contacts, ligand↔pocket distance).
- **apo `--pdb`** (no ligand): add `--pocket-ref-pdb <holo.pdb>` so the pocket is
  detected from a bound reference, and use `--global-token-set rmsd_e2e_pocket3`.

## Pocket auto-detection

Pocket residues are found as protein residues within a cutoff of the bound ligand
(default 0.5 nm) in the reference structure — no hardcoded residue numbers. `seed
metad --cv-set holo` detects the pocket from `--pdb`; `--cv-set apo` and apo
pretraining detect it from `--pocket-ref-pdb`.

## The λ-toggled ligand–pocket restraint

During `train`, `--ligand-pocket-restraint` adds a **flat-bottom** COM restraint
between the ligand and pocket (`--ligand-pocket-restraint-radius-nm`,
`--ligand-pocket-restraint-k`). It is **λ-toggled**:

- **ON during steering** — so the policy cannot reward-hack by dissociating the
  ligand into solvent (trivial "diversity").
- **OFF during production rollouts** — so the sampled dynamics stay unbiased.

For seeding, the metadynamics `holo` CV set applies its own always-on flat-bottom
restraint (`--lig-restraint-r0-nm`, `--lig-restraint-k`) to keep the ligand in the
pocket while sampling bound poses; `high-temp` offers a single-residue variant
(`--ligand-pocket-restraint --pocket-seq-pos`).

## Apo vs holo at a glance

| | apo | holo |
|---|---|---|
| `--ligand-smiles` | — | yes |
| seed `--cv-set` | `apo` (+`--pocket-ref-pdb`) | `holo` |
| pretrain token set | `rmsd_e2e_pocket3` (+`--pocket-ref-pdb`) | `rmsd_e2e_pocket_full` |
| train ligand restraint | — | `--ligand-pocket-restraint` |
