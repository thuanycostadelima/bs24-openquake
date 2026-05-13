"""
Plot BS24 Response Spectra - Cratonic vs NonCratonic
Reproduces the style of Figure D-16 in the BS24 appendix.

Usage:
    python plot_bs24_spectra.py

by T Costa de Lima, April 2026, Geoscience Australia
(thuany.costadelima@ga.gov.au)
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Make sure the GMM file is importable ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gmm'))
from bayless_somerville_2024 import (
    BaylessSomerville2024Cratonic,
    BaylessSomerville2024NonCratonic,
)
from openquake.hazardlib.imt import PGA, SA

# -----------------------------------------------------------------------
# Scenario parameters  (change these to edit the plot)
# -----------------------------------------------------------------------
MAGNITUDES  = [4.0, 5.0, 6.0, 7.0, 8.0]   # moment magnitudes
RRUP_LEFT   = 15.0    # km  - left panel (near source)
RRUP_RIGHT  = 150.0   # km  - right panel (regional)
ZTOR        = 0.0     # km  - surface rupture (Ztor=0)
DIP         = 45.0    # degrees
RAKE        = 90.0    # degrees (reverse)
WIDTH       = 15.0    # km fault width
VS30        = 760.0   # m/s reference rock
Z1PT0       = -999.   # no basin term (recommended for Australia)
RX          = -99.    # footwall (negative Rx)
RY0         = 0.0

# Spectral periods to evaluate
PERIODS = np.array([
    0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.075,
    0.100, 0.150, 0.200, 0.300, 0.400, 0.500, 0.750,
    1.000, 1.500, 2.000, 3.000, 4.000, 5.000, 7.500, 10.000
])

# Magnitude colours
MAG_COLORS = {
    4.0: '#1f77b4',   # blue
    5.0: '#ff7f0e',   # orange
    6.0: '#2ca02c',   # green
    7.0: '#d62728',   # red
    8.0: '#9467bd',   # purple
}

# -----------------------------------------------------------------------
# Helper function - build a context recarray for one scenario
# -----------------------------------------------------------------------
def make_ctx(mag, rrup):
    ctx = np.recarray(1, dtype=[
        ('mag',   float), ('rrup',  float), ('rjb',   float),
        ('rx',    float), ('ztor',  float), ('dip',   float),
        ('rake',  float), ('width', float), ('vs30',  float),
        ('z1pt0', float), ('ry0',   float),
    ])
    ctx.mag[:]   = mag
    ctx.rrup[:]  = rrup
    ctx.rjb[:]   = rrup      # approximate for footwall
    ctx.rx[:]    = RX
    ctx.ztor[:]  = ZTOR
    ctx.dip[:]   = DIP
    ctx.rake[:]  = RAKE
    ctx.width[:] = WIDTH
    ctx.vs30[:]  = VS30
    ctx.z1pt0[:] = Z1PT0
    ctx.ry0[:]   = RY0
    return ctx

# -----------------------------------------------------------------------
# Helper function - compute response spectrum for one GMM and scenario
# -----------------------------------------------------------------------
def compute_spectrum(gsim, mag, rrup):
    imts = [PGA()] + [SA(t) for t in PERIODS[PERIODS > 0.01]]
    all_periods = np.concatenate([[0.01], PERIODS[PERIODS > 0.01]])

    ctx  = make_ctx(mag, rrup)
    n    = len(imts)
    mean = np.zeros((n, 1))
    sig  = np.zeros((n, 1))
    tau  = np.zeros((n, 1))
    phi  = np.zeros((n, 1))
    gsim.compute(ctx, imts, mean, sig, tau, phi)

    # Convert from ln(g) to g
    sa_g = np.exp(mean[:, 0])
    return all_periods, sa_g, np.exp(mean[:, 0] + sig[:, 0]), np.exp(mean[:, 0] - sig[:, 0])

# -----------------------------------------------------------------------
# Build the figure  (2 panels: Rrup=15km and Rrup=150km)
# -----------------------------------------------------------------------
gsim_c  = BaylessSomerville2024Cratonic()
gsim_nc = BaylessSomerville2024NonCratonic()

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
fig.patch.set_facecolor('white')

panel_info = [
    (axes[0], RRUP_LEFT,  f'(a) R$_{{rup}}$ = {RRUP_LEFT:.0f} km'),
    (axes[1], RRUP_RIGHT, f'(b) R$_{{rup}}$ = {RRUP_RIGHT:.0f} km'),
]

for ax, rrup, panel_label in panel_info:

    for mag in MAGNITUDES:
        color = MAG_COLORS[mag]

        # Cratonic - dotted line 
        periods, sa_c, _, _ = compute_spectrum(gsim_c, mag, rrup)
        ax.loglog(periods, sa_c, color=color, lw=2.0,
                  linestyle=':', label=f'M{mag:.0f}' if ax is axes[0] else '')

        # NonCratonic - solid line
        periods, sa_nc, _, _ = compute_spectrum(gsim_nc, mag, rrup)
        ax.loglog(periods, sa_nc, color=color, lw=2.0,
                  linestyle='-', label='')

    # Formatting
    ax.set_xlabel('Period (s)', fontsize=12)
    ax.set_ylabel('RotD50 SA (g)', fontsize=12) if ax is axes[0] else None
    ax.set_title(panel_label, fontsize=12, loc='left', pad=8)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([1e-5, 2e0])
    ax.grid(True, which='both', alpha=0.3, lw=0.5)
    ax.tick_params(axis='both', which='both', labelsize=10)

    # Period axis ticks
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:g}'))
    ax.set_xticks([0.01, 0.1, 1.0, 10.0])

    # Scenario text box
    textstr = (f'Z$_{{tor}}$ = {ZTOR:.1f} km\n'
               f'V$_{{S30}}$ = {VS30:.0f} m/s\n'
               f'Footwall')
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))

# -----------------------------------------------------------------------
# Legend  — magnitude colours + line style for Cratonic/NonCratonic
# -----------------------------------------------------------------------
from matplotlib.lines import Line2D

mag_handles = [
    Line2D([0], [0], color=MAG_COLORS[m], lw=2, label=f'M {m:.0f}')
    for m in MAGNITUDES
]
style_handles = [
    Line2D([0], [0], color='k', lw=2, linestyle=':',  label='Cratonic'),
    Line2D([0], [0], color='k', lw=2, linestyle='-', label='NonCratonic'),
]
all_handles = mag_handles + style_handles

axes[0].legend(handles=all_handles, fontsize=9, loc='lower left',
               framealpha=0.9, ncol=1)

# -----------------------------------------------------------------------
# Title and save
# -----------------------------------------------------------------------
#fig.suptitle(
#    'Bayless and Somerville (2024) — Response Spectra\n'
#    'Cratonic (solid) vs NonCratonic (dotted), Vs30 = 760 m/s, Ztor = 0 km, Footwall',
#    fontsize=11, y=1.01
#)
plt.tight_layout()

outpath = 'bs24_response_spectra.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {outpath}')
