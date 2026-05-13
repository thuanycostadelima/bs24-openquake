## Step 6 — Quick sanity check (single scenario)

```bash
python - <<'EOF'
import numpy as np
from openquake.hazardlib.gsim.bayless_somerville_2024 import (
    BaylessSomerville2024Cratonic,
    BaylessSomerville2024NonCratonic,
)
from openquake.hazardlib.imt import PGA, SA

# Scenario: M6.5, Rrup=50km, Ztor=10km (reference depth), Vs30=760 m/s, footwall
ctx = np.recarray(1, dtype=[
    ('mag', float), ('rrup', float), ('rjb', float), ('rx', float),
    ('ztor', float), ('dip', float), ('rake', float), ('width', float),
    ('vs30', float), ('z1pt0', float), ('ry0', float),
])
ctx.mag[:] = 6.5; ctx.rrup[:] = 50.; ctx.rjb[:] = 50.; ctx.rx[:] = -50.
ctx.ztor[:] = 10.; ctx.dip[:] = 45.; ctx.rake[:] = 90.; ctx.width[:] = 10.
ctx.vs30[:] = 760.; ctx.z1pt0[:] = -999.; ctx.ry0[:] = 0.

imts = [PGA(), SA(0.2), SA(1.0), SA(3.0)]

for label, gsim_cls in [('Cratonic', BaylessSomerville2024Cratonic),
                         ('NonCratonic', BaylessSomerville2024NonCratonic)]:
    gsim = gsim_cls()
    n = len(imts)
    mean = np.zeros((n, 1)); sig = np.zeros_like(mean)
    tau = np.zeros_like(mean); phi = np.zeros_like(mean)
    gsim.compute(ctx, imts, mean, sig, tau, phi)
    print(f"\n{label} (M6.5, Rrup=50km, Ztor=10km, Vs30=760):")
    print(f"  {'IMT':<10} {'mean ln(g)':>12} {'sigma':>8} {'tau':>8} {'phi':>8}")
    for i, imt in enumerate(imts):
        print(f"  {str(imt):<10} {mean[i,0]:>12.4f} {sig[i,0]:>8.4f} "
              f"{tau[i,0]:>8.4f} {phi[i,0]:>8.4f}")

print("\nExpected: Cratonic PGA > NonCratonic PGA (higher source stress)")
print("Expected: fZtor = 0 at Ztor=10km (model centred at 10km, Rev2)")
EOF
```

---

## Step 7 — Generate response spectra figures

These scripts run from the repository root and save figures to the `plots/` folder.

**From the Python GMM:**
```bash
python benchmark/plots_py/plot_bs24_spectra_py.py
```
Output: `plots/bs24_response_spectra.png`

**From Fortran output** (requires running the benchmark first — see Step 8):
```bash
python benchmark/plots_py/plot_bs24_spectra_ft.py
```
Output: `plots/bs24_fortran_spectra.png`

**Side-by-side Fortran vs Python comparison:**
```bash
python benchmark/plots_py/plot_bs24_comparison.py
```
Output: `plots/bs24_spectra_comparison.png`

---

## Step 8 — Benchmarking against the Fortran reference

The Fortran subroutine `bs24_gmm.f` is not publicly available — contact Jeff Bayless (jeff.bayless@aecom.com) to obtain it. Once you have it, place it in the `benchmark/` folder and run:

```bash
cd benchmark

# Compile — the -ffixed-line-length-none flag is required because
# bs24_gmm.f contains lines longer than the standard 72-character limit
gfortran -O2 -ffixed-line-length-none -o bs24_run bs24_runner.f bs24_gmm.f

# Run Fortran to generate reference CSV
./bs24_run
# Output: benchmark/bs24_fortran_output.csv (14256 rows)

# Compare against Python
python benchmark_bs24.py
# Output: benchmark/benchmark_bs24_results.csv
#         benchmark/benchmark_bs24_plot.png
```

Expected result:
```
Total test cases : 1760
Median (mean SA): PASS 1760/1760  (100.0%)
Sigma:            PASS 1760/1760  (100.0%)
```

---

## Step 9 — Regenerate GEM verification CSV files

If there is any update to the model coefficients, regenerate the GEM CSV files with:

```bash
python tests/generate_gem_test_files.py
```

This overwrites:
- `tests/data/BS24_CRATONIC.csv`
- `tests/data/BS24_NONCRATONIC.csv`
- `tests/bayless_somerville_2024_test.py`

Then commit the updated files to GitHub.

## About `tests/bayless_somerville_2024_test.py`

This file is **not meant to be run on your local machine**. It is the GEM-format
verification test module, written for submission to the GEM OpenQuake engine
repository. It uses GEM's `BaseGSIMTestCase` framework which is only available
inside the full `oq-engine` source tree.

To verify correctness on your local machine, use the unit test suite instead:

    python benchmark/bs24_calc_suite.py -v

---


## Step 10 — Use in an OpenQuake PSHA job

In your gmpe logic tree XML:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">
  <logicTree logicTreeID="lt1">
    <logicTreeBranchSet
        uncertaintyType="gmpeModel"
        branchSetID="bs1"
        applyToTectonicRegionType="Stable Continental">
      <logicTreeBranch branchID="b1" weight="0.5">
        <uncertaintyModel>BaylessSomerville2024Cratonic</uncertaintyModel>
      </logicTreeBranch>
      <logicTreeBranch branchID="b2" weight="0.5">
        <uncertaintyModel>BaylessSomerville2024NonCratonic</uncertaintyModel>
      </logicTreeBranch>
    </logicTreeBranchSet>
  </logicTree>
</nrml>
```

Then run:
```bash
oq engine --run job.ini
```


---

## Troubleshooting

| Error | Fix |
|---|---|
| `ImportError: cannot import name 'BaylessSomerville2024Cratonic'` | Check Step 4 — `__init__.py` import missing |
| `SyntaxError when importing` | Bare text outside functions in the .py file |
| `KeyError: SA(0.0)` | Check `C_PGA = self.COEFFS[SA(0.01)]` not `PGA()` in `compute()` |
| `All means are zero` | GMM file not in the correct gsim directory (Step 3) |
| `Results differ from Fortran at T=0.75` | Expected — confirmed Fortran typo at that period |
| `gfortran: command not found` | `sudo apt-get install gfortran` or `brew install gcc` |
| `ModuleNotFoundError: openquake.hazardlib.tests` | The GEM test module needs the full oq-engine source tree |

---
