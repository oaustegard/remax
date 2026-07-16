"""Plot nemotron_1bit.csv results: two-panel layout (scifact + stsb).

Reads bench/results/nemotron_1bit.csv and generates bench/results/nemotron_1bit.png.
Layout: scifact (left) with r10_vs_float + ndcg10; stsb (right) with spearman.
X-axis: bytes_per_vec (log2 scale). Two visually distinct series families:
  - float32/MRL (circles, blue)
  - 1-bit/remax (squares, orange)

Usage
-----
::

    python bench/plot_nemotron_1bit.py [--csv <path>] [--out <path>] [--selftest]

Examples
--------
::

    # Standard run (reads bench/results/nemotron_1bit.csv)
    python bench/plot_nemotron_1bit.py

    # Custom input/output paths
    python bench/plot_nemotron_1bit.py --csv /tmp/custom.csv --out /tmp/plot.png

    # Smoke test on synthetic data
    python bench/plot_nemotron_1bit.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_csv(csv_path):
    """Parse nemotron_1bit.csv into structured dicts by dataset.

    Returns
    -------
    dict
        {"scifact": [(method, bytes, r10_vs_float, ndcg10, ...], ...},
         "stsb": [...]}
    """
    data = {"scifact": [], "stsb": []}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset']
            if dataset not in data:
                continue

            method = row['method']
            bytes_per_vec = float(row['bytes_per_vec'])

            entry = {
                'method': method,
                'bytes_per_vec': bytes_per_vec,
                'dims': int(row['dims']),
                'k_stack': int(row['k_stack']),
                'compression_x': float(row['compression_x']),
            }

            if dataset == 'scifact':
                r10 = row.get('r10_vs_float', '').strip()
                ndcg = row.get('ndcg10', '').strip()
                entry['r10_vs_float'] = float(r10) if r10 else None
                entry['ndcg10'] = float(ndcg) if ndcg else None
            else:  # stsb
                spear = row.get('spearman', '').strip()
                entry['spearman'] = float(spear) if spear else None

            data[dataset].append(entry)

    return data


def classify_method(method_id):
    """Classify a method as float32/MRL or 1-bit/remax.

    Returns
    -------
    tuple
        (family, is_float32, label) where family in {'float', 'remax'}
    """
    if method_id.startswith('f32'):
        return ('float', True, method_id)
    else:  # bit1_*
        return ('remax', False, method_id)


def plot_nemotron(data, out_path):
    """Generate two-panel plot: scifact (left) and stsb (right).

    Parameters
    ----------
    data : dict
        Output from parse_csv()
    out_path : str or Path
        Output PNG file path
    """
    # Colorblind-safe palette
    color_float = '#4053d3'  # blue
    color_remax = '#dd8021'  # orange

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # ---- SciFact panel (left) ----
    ax_sci = axes[0]
    scifact_rows = data['scifact']

    # Separate by family
    float_rows = [r for r in scifact_rows if r['method'].startswith('f32')]
    remax_rows = [r for r in scifact_rows if r['method'].startswith('bit1')]

    # Get f32_2048 reference values
    f32_2048_row = next((r for r in float_rows if r['method'] == 'f32_2048'), None)
    if f32_2048_row:
        r10_ref = f32_2048_row['r10_vs_float']
        ndcg_ref = f32_2048_row['ndcg10']
    else:
        r10_ref = ndcg_ref = None

    # Plot float32/MRL series (circles)
    if float_rows:
        bytes_vals = [r['bytes_per_vec'] for r in float_rows]
        r10_vals = [r['r10_vs_float'] for r in float_rows]
        ndcg_vals = [r['ndcg10'] for r in float_rows]

        ax_sci.plot(bytes_vals, r10_vals, 'o-', color=color_float, label='float32/MRL (R@10)',
                    linewidth=1.5, markersize=6, alpha=0.8)
        ax_sci.plot(bytes_vals, ndcg_vals, 'o--', color=color_float, label='float32/MRL (nDCG@10)',
                    linewidth=1.5, markersize=6, alpha=0.6)

        # Label each point
        for r in float_rows:
            b = r['bytes_per_vec']
            r10 = r['r10_vs_float']
            ndcg = r['ndcg10']
            ax_sci.annotate(r['method'], (b, r10), fontsize=7, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
            ax_sci.annotate('', (b, ndcg), fontsize=7, ha='center', va='bottom')

    # Plot 1-bit/remax series (squares)
    if remax_rows:
        bytes_vals = [r['bytes_per_vec'] for r in remax_rows]
        r10_vals = [r['r10_vs_float'] for r in remax_rows]
        ndcg_vals = [r['ndcg10'] for r in remax_rows]

        ax_sci.plot(bytes_vals, r10_vals, 's-', color=color_remax, label='1-bit/remax (R@10)',
                    linewidth=1.5, markersize=6, alpha=0.8)
        ax_sci.plot(bytes_vals, ndcg_vals, 's--', color=color_remax, label='1-bit/remax (nDCG@10)',
                    linewidth=1.5, markersize=6, alpha=0.6)

        # Label each point
        for r in remax_rows:
            b = r['bytes_per_vec']
            r10 = r['r10_vs_float']
            ndcg = r['ndcg10']
            ax_sci.annotate(r['method'], (b, r10), fontsize=7, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
            ax_sci.annotate('', (b, ndcg), fontsize=7, ha='center', va='bottom')

    # Reference lines
    if r10_ref is not None:
        ax_sci.axhline(y=r10_ref, color='gray', linestyle=':', alpha=0.5, label='float32 full (R@10)')
    if ndcg_ref is not None:
        ax_sci.axhline(y=ndcg_ref, color='gray', linestyle=':', alpha=0.5)

    ax_sci.set_xscale('log', base=2)
    ax_sci.set_xlabel('bytes per vector (log scale)', fontsize=10)
    ax_sci.set_ylabel('Metric value', fontsize=10)
    ax_sci.set_title('SciFact (R@10 + nDCG@10)', fontsize=11, fontweight='bold')
    ax_sci.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax_sci.spines['top'].set_visible(False)
    ax_sci.spines['right'].set_visible(False)
    ax_sci.legend(fontsize=8, loc='best')

    # ---- STS-B panel (right) ----
    ax_stsb = axes[1]
    stsb_rows = data['stsb']

    float_rows_stsb = [r for r in stsb_rows if r['method'].startswith('f32')]
    remax_rows_stsb = [r for r in stsb_rows if r['method'].startswith('bit1')]

    # Get f32_2048 reference
    f32_2048_stsb = next((r for r in float_rows_stsb if r['method'] == 'f32_2048'), None)
    spear_ref = f32_2048_stsb['spearman'] if f32_2048_stsb else None

    # Plot float32/MRL (circles)
    if float_rows_stsb:
        bytes_vals = [r['bytes_per_vec'] for r in float_rows_stsb]
        spear_vals = [r['spearman'] for r in float_rows_stsb]

        ax_stsb.plot(bytes_vals, spear_vals, 'o-', color=color_float, label='float32/MRL',
                     linewidth=1.5, markersize=6, alpha=0.8)

        for r in float_rows_stsb:
            b = r['bytes_per_vec']
            s = r['spearman']
            ax_stsb.annotate(r['method'], (b, s), fontsize=7, ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    # Plot 1-bit/remax (squares)
    if remax_rows_stsb:
        bytes_vals = [r['bytes_per_vec'] for r in remax_rows_stsb]
        spear_vals = [r['spearman'] for r in remax_rows_stsb]

        ax_stsb.plot(bytes_vals, spear_vals, 's-', color=color_remax, label='1-bit/remax',
                     linewidth=1.5, markersize=6, alpha=0.8)

        for r in remax_rows_stsb:
            b = r['bytes_per_vec']
            s = r['spearman']
            ax_stsb.annotate(r['method'], (b, s), fontsize=7, ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    # Reference line
    if spear_ref is not None:
        ax_stsb.axhline(y=spear_ref, color='gray', linestyle=':', alpha=0.5, label='float32 full')

    ax_stsb.set_xscale('log', base=2)
    ax_stsb.set_xlabel('bytes per vector (log scale)', fontsize=10)
    ax_stsb.set_ylabel('Spearman ρ', fontsize=10)
    ax_stsb.set_title('STS-B (Spearman correlation)', fontsize=11, fontweight='bold')
    ax_stsb.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax_stsb.spines['top'].set_visible(False)
    ax_stsb.spines['right'].set_visible(False)
    ax_stsb.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_synthetic_csv():
    """Generate a minimal synthetic CSV for selftest.

    Returns
    -------
    str
        CSV content as string
    """
    rows = [
        ['dataset', 'method', 'dims', 'k_stack', 'bytes_per_vec', 'compression_x', 'r10_vs_float', 'ndcg10', 'spearman'],
        # SciFact rows
        ['scifact', 'f32_2048', '2048', '1', '8192', '1.0', '1.0', '1.0', ''],
        ['scifact', 'f32_mrl256', '256', '1', '1024', '8.0', '0.95', '0.93', ''],
        ['scifact', 'bit1_2048', '2048', '1', '256', '32.0', '0.78', '0.75', ''],
        ['scifact', 'bit1_stack4', '2048', '4', '1024', '8.0', '0.92', '0.90', ''],
        # STS-B rows
        ['stsb', 'f32_2048', '2048', '1', '8192', '1.0', '', '', '0.87'],
        ['stsb', 'f32_mrl256', '256', '1', '1024', '8.0', '', '', '0.81'],
        ['stsb', 'bit1_2048', '2048', '1', '256', '32.0', '', '', '0.65'],
        ['stsb', 'bit1_stack4', '2048', '4', '1024', '8.0', '', '', '0.79'],
    ]

    out = io.StringIO()
    w = csv.writer(out)
    for row in rows:
        w.writerow(row)
    return out.getvalue()


def test_selftest(csv_content, out_path):
    """Run synthetic data through the plotting pipeline.

    Parameters
    ----------
    csv_content : str
        CSV text
    out_path : Path
        Path to write PNG

    Returns
    -------
    bool
        True if all checks pass
    """
    # Write synthetic CSV to temp
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        tmp_csv = Path(f.name)

    try:
        # Parse and plot
        data = parse_csv(tmp_csv)

        # Verify structure
        assert 'scifact' in data, "Missing scifact dataset"
        assert 'stsb' in data, "Missing stsb dataset"
        assert len(data['scifact']) > 0, "No scifact rows"
        assert len(data['stsb']) > 0, "No stsb rows"

        # Check that f32_2048 has full value
        f32_2048_sci = next((r for r in data['scifact'] if r['method'] == 'f32_2048'), None)
        assert f32_2048_sci is not None, "f32_2048 missing from scifact"
        assert f32_2048_sci['r10_vs_float'] == 1.0, f"f32_2048 R@10 != 1.0: {f32_2048_sci['r10_vs_float']}"

        # Check bytes_per_vec values make sense
        for r in data['scifact']:
            assert r['bytes_per_vec'] > 0, f"Invalid bytes_per_vec: {r['bytes_per_vec']}"

        # Generate plot
        plot_nemotron(data, out_path)

        # Verify PNG was created
        assert out_path.exists(), f"PNG not created at {out_path}"
        assert out_path.stat().st_size > 0, f"PNG is empty: {out_path}"

        return True

    finally:
        tmp_csv.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default='bench/results/nemotron_1bit.csv',
                   help='Input CSV path (default: bench/results/nemotron_1bit.csv)')
    p.add_argument('--out', type=Path, default='bench/results/nemotron_1bit.png',
                   help='Output PNG path (default: bench/results/nemotron_1bit.png)')
    p.add_argument('--selftest', action='store_true',
                   help='Run smoke test on synthetic data')
    args = p.parse_args()

    if args.selftest:
        # Generate synthetic CSV and test
        csv_content = make_synthetic_csv()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'test.png'

            try:
                success = test_selftest(csv_content, out_path)
                if success:
                    print('PASS: Synthetic CSV plot generated successfully')
                    return 0
            except Exception as e:
                print(f'FAIL: {e}', file=sys.stderr)
                return 1

    else:
        # Normal mode: read nemotron_1bit.csv and generate plot
        if not args.csv.exists():
            print(f'ERROR: CSV file not found: {args.csv}', file=sys.stderr)
            return 1

        try:
            data = parse_csv(args.csv)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            plot_nemotron(data, args.out)
            print(f'Plot written to {args.out}')
            return 0
        except Exception as e:
            print(f'ERROR: {e}', file=sys.stderr)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
