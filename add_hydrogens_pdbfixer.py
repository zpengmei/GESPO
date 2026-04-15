#!/usr/bin/env python3
"""
Alternative approach: Use pdbfixer (lighter weight than full OpenMM) to add hydrogens.
This should work if pdbfixer is available.
"""
import sys

try:
    from pdbfixer import PDBFixer
    from simtk.openmm.app import PDBFile
    HAS_PDBFIXER = True
except ImportError:
    HAS_PDBFIXER = False

def add_hydrogens_pdbfixer(input_pdb, output_pdb):
    """Add hydrogens using pdbfixer."""
    print(f"Using pdbfixer to add hydrogens...")
    
    fixer = PDBFixer(filename=input_pdb)
    
    # Remove non-protein residues
    fixer.removeHeterogens(keepWater=False)
    
    # Find and add missing atoms (including hydrogens)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    # Add hydrogens at pH 7.0
    fixer.addMissingHydrogens(7.0)
    
    # Save
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    print(f"✓ Saved to: {output_pdb}")
    return True

def main():
    input_pdb = "/beagle3/dinner/Shared_nhan_zihan/chignolin/trans_1130/5awl.pdb"
    output_pdb = "/beagle3/dinner/Shared_nhan_zihan/chignolin/trans_1130/5awl_with_H.pdb"
    
    if not HAS_PDBFIXER:
        print("ERROR: pdbfixer is not installed.")
        print("\nInstallation options:")
        print("  1. conda install -c conda-forge pdbfixer")
        print("  2. pip install pdbfixer")
        print("\nAlternative: Use chignolin.pdb which already has the correct structure")
        sys.exit(1)
    
    add_hydrogens_pdbfixer(input_pdb, output_pdb)
    
    # Verify
    import mdtraj as md
    processed = md.load(output_pdb)
    ref = md.load("/beagle3/dinner/Shared_nhan_zihan/chignolin/trans_1130/chignolin.pdb")
    
    proc_heavy = [a for a in processed.topology.atoms if a.element.symbol != 'H']
    ref_heavy = [a for a in ref.topology.atoms if a.element.symbol != 'H']
    
    print(f"\nVerification:")
    print(f"  Processed atoms: {processed.n_atoms} (heavy: {len(proc_heavy)})")
    print(f"  Reference atoms: {ref.n_atoms} (heavy: {len(ref_heavy)})")
    
    if len(proc_heavy) == len(ref_heavy):
        print(f"\n✓ SUCCESS! Use: --pdb {output_pdb}")
    else:
        print(f"\n✗ Atom count mismatch - may need manual adjustment")

if __name__ == "__main__":
    main()
