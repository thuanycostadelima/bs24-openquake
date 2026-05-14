import os
"""
BS24 Response Spectra Comparison: Fortran vs Python
=====================================================
Reads benchmark_bs24_results.csv and plots response spectra
for both Fortran and Python side by side, for a selection of
representative scenarios.

Usage:
    python plot_bs24_comparison.py

Requires:
    - benchmark_bs24_results.csv  (from benchmark_bs24.py)
    - matplotlib, pandas, numpy
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------------------------------------------------
# Load benchmark results
# -----------------------------------------------------------------------
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'benchmark_bs24_results.csv'))

# -----------------------------------------------------------------------
# Select scenarios to plot
# Showing 4 panels: 2 magnitudes x 2 distances, Vs30=760, Ztor=0, Cratonic
# -----------------------------------------------------------------------
SCENARIOS = [
    # (CratFlag, mag, rrup, vs30, ztor, panel_label)
    (1, 5.5,  50., 760., 0., '(a) Cratonic  M5.5  R=50 km'),
    (1, 6.5,  50., 760., 0., '(b) Cratonic  M6.5  R=50 km'),
    (1, 5.5, 100., 760., 0., '(c) Cratonic  M5.5  R=100 km'),
    (1, 6.5, 100., 760., 0., '(d) Cratonic  M6.5  R=100 km'),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
fig.patch.set_facecolor('white')
axes = axes.flatten()

for ax, (crat, mag, rrup, vs30, ztor, label) in zip(axes, SCENARIOS):

    sub = df[
        (df['CratFlag'] == crat) &
        (np.isclose(df['mag'],  mag,  atol=0.01)) &
        (np.isclose(df['rrup'], rrup, atol=0.1)) &
        (np.isclose(df['vs30'], vs30, atol=1.)) &
        (np.isclose(df['ztor'], ztor, atol=0.1))
    ].copy()

    if sub.empty:
        ax.set_title(f'{label}\n(no data)', fontsize=10)
        continue

    # Sort by period, exclude PGA (period=0) for the spectra plot
    sub = sub[sub['period'] > 0].sort_values('period')

    # Convert Fortran ln(cm/s/s) -> g
    fort_g = np.exp(sub['fort_lng'].values)
    py_g   = np.exp(sub['py_lng'].values)
    periods = sub['period'].values

    # --- Main spectra panel ---
    ax.loglog(periods, fort_g, 'b-',  lw=2.5, label='Fortran')
    ax.loglog(periods, py_g,   'r--', lw=1.5, label='Python')

    ax.set_xlabel('Period (s)', fontsize=11)
    ax.set_ylabel('RotD50 SA (g)', fontsize=11)
    ax.set_title(label, fontsize=10, loc='left')
    ax.set_xlim([0.01, 10])
    ax.grid(True, which='both', alpha=0.3, lw=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xticks([0.01, 0.1, 1.0, 10.0])
    ax.legend(fontsize=9)

    # Inset: residual (py - fort) in ln units
    ax_in = ax.inset_axes([0.55, 0.05, 0.42, 0.35])
    residuals = sub['diff_mean'].values
    ax_in.semilogx(periods, residuals, 'k-', lw=1.2)
    ax_in.axhline(0,      color='k',  lw=0.6)
    ax_in.axhline( 0.001, color='r',  lw=0.8, ls='--')
    ax_in.axhline(-0.001, color='r',  lw=0.8, ls='--')
    ax_in.set_xlim([0.01, 10])
    ax_in.set_ylabel('Δln', fontsize=7)
    ax_in.tick_params(labelsize=6)
    ax_in.set_xticks([0.01, 0.1, 1.0, 10.0])
    ax_in.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax_in.grid(True, which='both', alpha=0.2)
    max_res = np.abs(residuals).max()
    ax_in.set_title(f'max|Δln|={max_res:.5f}', fontsize=6, pad=2)

fig.suptitle(
    'BS24 Response Spectra — Fortran (blue) vs Python (red dashed)\n'
    'Vs30 = 760 m/s, Ztor = 0 km, Footwall, Cratonic\n'
    'Insets show Python − Fortran residuals in ln units (red dashed = ±0.001)',
    fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'plots', 'bs24_spectra_comparison.png'), dpi=150,
            bbox_inches='tight', facecolor='white')
print('Saved: bs24_spectra_comparison.png')
