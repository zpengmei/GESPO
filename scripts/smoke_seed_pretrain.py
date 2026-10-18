"""Small CPU workflow check using OpenMM's bundled test protein, not a benchmark."""
import argparse
from pathlib import Path
import sys

import openmm.app as app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gespo.seed.methods.metad import run_metad
from gespo.training.pretrain_torsion_ar import pretrain_torsion_ar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, help='A new output directory.')
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    source = str(Path(app.__file__).parent / 'data/test.pdb')
    run_metad(pdb=source, out_dir=str(out / 'seed'), cv_set='folding', explicit=False,
              platform_name='CPU', production_ns=.002, equilibration_ns=0, save_ps=.5)
    pretrain_torsion_ar(source_path=str(out / 'seed'), out_path=str(out / 'policy.pt'),
                        pdb_path=str(out / 'seed/simulation_topology.pdb'),
                        global_token_set='rmsd_e2e_rg', epochs=1, batch_size=4,
                        device='cpu', d_model=32, n_layers=1, n_heads=2, kmix=2)
    print('CPU seed-to-pretraining smoke test completed; no scientific convergence is implied.')


if __name__ == '__main__':
    main()
