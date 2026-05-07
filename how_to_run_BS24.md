This is an OpenQuake hazardlib implementation of the Bayless and Somerville (2024) Ground Motion Model for Australia.
**Bayless and Somerville (2024) — Australian Ground Motion Model**  
Implementation by Thuany Costa de Lima (thuany.costadelima@ga.gov.au), May 2026


[![GitHub](https://img.shields.io/badge/GitHub-thuanycostadelima%2Fbs24--openquake-blue?logo=github)](https://github.com/thuanycostadelima/bs24-openquake)
---
## Repository Contents:

```
bayless_somerville_2024.py       main GMM implementation (OpenQuake GSIM)
test_bayless_somerville_2024.py  the test suite (26 tests)
bs24_runner.f                    fortran driver for benchmarking (requires `bs24_gmm.f` from authors)
benchmark_bs24.py                benchmark script to compare Fortran and Python output
plot_bs24_spectra.py             response spectra plot from Python GMM
plot_bs24_fortran_spectra.py     response spectra plot from Fortran output
plot_bs24_comparison.py          side-by-side Fortran vs Python comparison plot
```
> **Note:** The Fortran subroutine `bs24_gmm.f` (authored by Jeff Bayless, AECOM) is not included in this repository as it has not been publicly released. `bs24_runner.f` is a driver program written by myself that calls `bs24_gmm.f` — to use it you will need to obtain `bs24_gmm.f` directly from the authors.

---
# How to Install and Run the BS24 GMM in OpenQuake
## Installation

### 1. Install OpenQuake

```bash
pip install openquake.engine
```
---
## Step 2 — Find your OpenQuake gsim directory

```bash
python -c "import openquake.hazardlib.gsim as g; import os; print(os.path.dirname(g.__file__))"
```

This will print something like:
```
/home/user/openquake/hazardlib/gsim
```
or
```
/usr/local/lib/python3.x/dist-packages/openquake/hazardlib/gsim
```

That is your **gsim directory**. Keep note of it.

---

## Step 3 — Copy the GMM file into OpenQuake

```bash
cp bayless_somerville_2024.py /path/to/openquake/hazardlib/gsim/
```

---

## Step 4 — Register the model in OpenQuake's `__init__.py`

Open the `__init__.py` in the gsim directory:

```bash
vi /path/to/openquake/hazardlib/gsim/__init__.py
```

Add these two lines in the imports section (e.g. near other models):

```python
from openquake.hazardlib.gsim.bayless_somerville_2024 import (
    BaylessSomerville2024Cratonic,
    BaylessSomerville2024NonCratonic,
)
```
---
## Step 5 — Run the test suite

```bash
python test_bayless_somerville_2024.py -v
```

Expected output:
```
Ran 26 tests in 0.XXXs
OK
```
---
## Step 6 — Quick sanity check (single scenario)

```bash
python - <<'EOF'
import numpy as np
from openquake.hazardlib.gsim.bayless_somerville_2024 import (BaylessSomerville2024Cratonic, BaylessSomerville2024NonCratonic)
from openquake.hazardlib.imt import PGA, SA

# Scenario: M6.5, Rrup=50km, Ztor=5km, Vs30=760 m/s, footwall
ctx = np.recarray(1, dtype=[
    ('mag',   float), ('rrup',  float), ('rjb',   float),
    ('rx',    float), ('ztor',  float), ('dip',   float),
    ('rake',  float), ('width', float), ('vs30',  float),
    ('z1pt0', float), ('ry0',   float),
])
ctx.mag[:]   = 6.5
ctx.rrup[:]  = 50.
ctx.rjb[:]   = 50.
ctx.rx[:]    = -50.    # negative = footwall, no hanging wall effect
ctx.ztor[:]  = 5.
ctx.dip[:]   = 45.
ctx.rake[:]  = 90.
ctx.width[:] = 10.
ctx.vs30[:]  = 760.    # reference rock
ctx.z1pt0[:] = -999.   # no basin term
ctx.ry0[:]   = 0.

imts = [PGA(), SA(0.2), SA(1.0), SA(3.0)]

for label, gsim_cls in [('Cratonic',    BaylessSomerville2024Cratonic),
                         ('NonCratonic', BaylessSomerville2024NonCratonic)]:
    gsim = gsim_cls()
    n = len(imts)
    mean = np.zeros((n, 1))
    sig  = np.zeros((n, 1))
    tau  = np.zeros((n, 1))
    phi  = np.zeros((n, 1))
    gsim.compute(ctx, imts, mean, sig, tau, phi)
    print(f"\n{label} (M6.5, Rrup=50km, Ztor=5km, Vs30=760):")
    print(f"  {'IMT':<10} {'mean ln(g)':>12} {'sigma':>8} {'tau':>8} {'phi':>8}")
    for i, imt in enumerate(imts):
        print(f"  {str(imt):<10} {mean[i,0]:>12.4f} {sig[i,0]:>8.4f} "
              f"{tau[i,0]:>8.4f} {phi[i,0]:>8.4f}")
EOF
```

Expected behaviour:
- Cratonic PGA > NonCratonic PGA (higher source stress)
- Both models: sigma ~ 0.69–0.77 at short periods
- mean ln(g) in range roughly -5 to -2 for this scenario

---
## Step 7 — Benchmarking against the Fortran reference

  The Fortran subroutine `bs24_gmm.f` is not publicly available — contact
  Jeff Bayless (jeff.bayless@aecom.com) to obtain it. Once you have it:

  ```bash
  # Compile (both files must be in the same directory)
  gfortran -O2 -ffixed-line-length-none -o bs24_run bs24_runner.f bs24_gmm.f

  # Run Fortran to generate reference CSV
  ./bs24_run

  # Compare against Python
  python benchmark_bs24.py
  ```

  Expected result: 1760/1760 test cases pass on both mean SA and sigma.
---
## Step 8 — Use in an OpenQuake PSHA job

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

## Implementation notes

  The following details were verified against the Fortran reference and confirmed by Jeff Bayless (personal communication, May 2026):

  - 1. Rock PGA reference uses SA(0.01):  The nonlinear site term requires a reference PGA at Vs30=760 m/s. The Fortran
  computes this at T=0.01 sec, not T=0 (PGA). The Python implementation matches: 
  ```
  C_PGA = self.COEFFS[SA(0.01)]
  ```
  This only affects results at Vs30 different from 760 m/s.

  - 2. c4M lower anchor = 2.0. For M < M2 = 5.0, the near-source saturation taper uses a lower anchor of 2.0
  (not 1.0 as in the original ASK14). This was an intentional choice by Jeff Bayless based on scaling for very small magnitudes at short distances, and is
  not documented in the written equations of Abrahamson et al. (2014).
  ```
  c4m = c4 - (c4 - 2.0) * (5.0 - mag)   for 4 <= M < 5
  c4m = 2.0                               for M < 4
  ```

  - 3. Geometric spreading frozen at M2 for M < M2. For M < M2 = 5.0, geometric spreading is evaluated at M2, not the actual
  magnitude. This is consistent with the Fortran and the OpenQuake ASK14 implementation.

  - 4. Basin term threshold = 0.65 sec. The basin depth term fZ1.0 is suppressed for T < 0.65 sec, following BSSA14
  Equation 9. The Fortran used 0.75 sec which was a typo.

  - 5. Basin term off by default at all sites `ctx.z1pt0 = -999` to suppress the basin term (recommended for Australia until Z1.0 measurements are collected):
  ```
  ctx.z1pt0[:] = -999.
  ```

  - 6. hwflag parameter: Pass `hwflag=0` to suppress the hanging wall term for point source scenarios:
  ```
  gsim = BaylessSomerville2024Cratonic(hwflag=0)
  ```
---

## Troubleshooting:

  | Error | Fix |
  |---|---|
  | `ImportError: cannot import name 'BaylessSomerville2024Cratonic'` | Check Step 3 — `__init__.py` import missing |
  | `SyntaxError when importing` | Check for bare text outside functions in the .py file |
  | `KeyError: SA(0.0)` | Make sure `C_PGA = self.COEFFS[SA(0.01)]` not `PGA()` in `compute()` |
  | `All means are zero` | GMM file not in the correct gsim directory (Step 2) |
  | `Results differ from Fortran at T=0.75` | Expected — Fortran has a confirmed typo at this period |
  | `gfortran: command not found` | `sudo apt-get install gfortran` or `brew install gcc` |

---

## Range of applicability

  M 4.0–8.0, Rrup 0–300 km, T 0.01–10 sec, Vs30 150–1500 m/s.
  Results outside these ranges are extrapolations.

---

## References

  Bayless J. and P. Somerville (2024). An Updated Ground Motion Model for Australia Developed Using Broadband Ground Motion Simulations. *Proc. AEES 2024 National Conference*, 21–23 November 2024, Adelaide, South Australia.

  Abrahamson N.A., Silva W.J., and Kamai R. (2014). Summary of the ASK14 Ground Motion Relation for Active Crustal Regions. *Earthquake Spectra* 30, 1025–1055.

  Boore D.M., Stewart J.P., Seyhan E., and Atkinson G.M. (2014). NGA-West2 Equations for Predicting PGA, PGV, and 5% Damped PSA for Shallow Crustal Earthquakes. *Earthquake Spectra* 30, 1057–1085.

  Donahue J.L. and Abrahamson N.A. (2014). Simulation-Based Hanging Wall Effects. *Earthquake Spectra* 30, 1269–1284.

  Al Atik L. (2015). NGA-East: Ground-Motion Standard Deviation Models for Central and Eastern North America. PEER Report 2015/07.

  Somerville P.G. et al. (2009). Source and Ground Motion Models of Australian Earthquakes. *Proc. AEES 2009 Annual Conference*, Newcastle.
