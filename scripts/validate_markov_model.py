"""Analyze a saved coarse transition matrix; does not certify physical kinetics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gespo.analysis.markov_validation import analyze_transition_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transition', required=True, help='Row-stochastic .npy matrix.')
    parser.add_argument('--reactant', required=True, help='Comma-separated state indices.')
    parser.add_argument('--product', required=True, help='Comma-separated state indices.')
    parser.add_argument('--out', required=True, help='Output .npz path.')
    args = parser.parse_args()
    endpoints = lambda value: np.asarray([int(i) for i in value.split(',')], dtype=int)
    result = analyze_transition_model(np.load(args.transition, allow_pickle=False),
                                      endpoints(args.reactant), endpoints(args.product))
    np.savez_compressed(args.out, **result)
    print(json.dumps({
        'scope': 'Fitted Markov model only; physical Markov and sampling assumptions remain unvalidated.',
        'detailed_balance_residual': result['detailed_balance_residual'],
        'backward_complement_error': result['backward_complement_error'],
        'output': args.out,
    }, indent=2))


if __name__ == '__main__':
    main()
