## Overview

This repository contains a Python implementation of the Bayless and Somerville (2024; BS24) GMM for Australia, developed for use within the [OpenQuake hazardlib](https://github.com/gem/oq-engine) framework.

BS24 is an update to the Somerville et al. (2009; Sea09) GMM, adopting the ASK14 functional form (Abrahamson et al. 2014) and incorporating new broadband ground motion simulations and recorded data from Australian earthquakes. Separate **Cratonic** and **NonCratonic** versions are provided, reflecting differences in source and crustal structure between regions.

The implementation was verified by direct comparison against the Fortran reference subroutine provided by Jeff Bayless (AECOM), achieving agreement to within floating-point precision across 1760 test cases.

---

## Model Description

The median model follows Equation D-2 of the BS24 appendix:

  ln(RotD50) = fM + fP + fZtor + fS + fZ1.0 + fHW


| Component | Description | Source |
|---|---|---|
| fM | Magnitude scaling | ASK14 polynomial (Eq. D-3) |
| fP | Path scaling — geometric spreading + anelastic Q | ASK14 form (Eq. D-4) |
| fZtor | Depth to top of rupture scaling | New polynomial (Eq. D-5) |
| fS | Vs30 site amplification (linear + nonlinear) | Boore et al. (2014) |
| fZ1.0 | Basin depth scaling | Boore et al. (2014) |
| fHW | Hanging wall effects | Donahue & Abrahamson (2014) via ASK14 |

Aleatory variability follows Al Atik (2015): `sigma = sqrt(tau^2 + phi^2)`

Output: ln(RotD50) in units of **g** (OpenQuake standard).

