"""Recompute coverage curves from exported frame coordinates and production budgets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gespo.analysis.coverage import coverage_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='method,replicate,budget_ns,cv_x,cv_y columns.')
    parser.add_argument('--x-range', nargs=2, type=float, required=True)
    parser.add_argument('--y-range', nargs=2, type=float, required=True)
    parser.add_argument('--grid-size', type=int, default=60)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    source, out = Path(args.csv), Path(args.out_dir)
    groups = {}
    with source.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['method'], row['replicate']
            groups.setdefault(key, []).append((float(row['budget_ns']), float(row['cv_x']), float(row['cv_y'])))
    if not groups:
        raise ValueError('No benchmark frames found.')
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    exclusions = {}
    with (out / 'coverage.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'replicate', 'budget_ns', 'occupied_bins'])
        for (method, replicate), rows in sorted(groups.items()):
            times, counts, excluded = coverage_curve([r[0] for r in rows], [r[1:] for r in rows],
                                                       args.x_range, args.y_range, args.grid_size)
            writer.writerows((method, replicate, time, count) for time, count in zip(times, counts))
            ax.step(times, counts, where='post', label=f'{method} / {replicate}')
            exclusions[f'{method}/{replicate}'] = excluded
    ax.set(xlabel='Aggregate unbiased production MD (ns)', ylabel='Occupied CV bins')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'coverage.png', dpi=180)
    plt.close(fig)
    (out / 'provenance.json').write_text(json.dumps(dict(
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        x_range=args.x_range, y_range=args.y_range, grid_size=args.grid_size,
        excluded_frames=exclusions,
        scope='Coverage only. Input provenance, equal-budget design, and independent replicates must be checked separately.'
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
