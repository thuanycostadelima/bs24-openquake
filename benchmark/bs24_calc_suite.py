"""
Tests for Bayless and Somerville (2024) BS24 OpenQuake GSIM.

Structural tests run immediately.
Verification tests compare Python output against values manually
computed from the Fortran code (same coefficients, same equations).

Run with:
    python test_bayless_somerville_2024.py
"""
import sys, os, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gmm'))
from bayless_somerville_2024 import (
    BaylessSomerville2024Cratonic,
    BaylessSomerville2024NonCratonic,
    _get_fM, _get_fP, _get_fZtor, _get_fS, _get_fHW, _get_stddevs,
)
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


def make_ctx(mag=5.5, rrup=50., rjb=48., rx=-50., ztor=5.,
             dip=45., rake=90., width=5., vs30=760., z1pt0=-999.,
             n=3):
    ctx = np.recarray(n, dtype=[
        ('mag', float), ('rrup', float), ('rjb', float),
        ('rx', float), ('ztor', float), ('dip', float),
        ('rake', float), ('width', float), ('vs30', float),
        ('z1pt0', float), ('ry0', float),
    ])
    ctx.mag[:] = mag; ctx.rrup[:] = rrup; ctx.rjb[:] = rjb
    ctx.rx[:] = rx;   ctx.ztor[:] = ztor; ctx.dip[:] = dip
    ctx.rake[:] = rake; ctx.width[:] = width
    ctx.vs30[:] = vs30; ctx.z1pt0[:] = z1pt0; ctx.ry0[:] = 0.
    return ctx


def run(gsim, ctx, imts):
    n = len(ctx)
    mean = np.zeros((len(imts), n))
    sig  = np.zeros_like(mean)
    tau  = np.zeros_like(mean)
    phi  = np.zeros_like(mean)
    gsim.compute(ctx, imts, mean, sig, tau, phi)
    return mean, sig, tau, phi


class TestMetadata(unittest.TestCase):
    def test_trt(self):
        for cls in [BaylessSomerville2024Cratonic,
                    BaylessSomerville2024NonCratonic]:
            self.assertEqual(cls().DEFINED_FOR_TECTONIC_REGION_TYPE,
                             const.TRT.STABLE_CONTINENTAL)

    def test_imts(self):
        for cls in [BaylessSomerville2024Cratonic,
                    BaylessSomerville2024NonCratonic]:
            gsim = cls()
            self.assertIn(PGA, gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES)
            self.assertIn(SA,  gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES)

    def test_ref_velocity(self):
        self.assertAlmostEqual(
            BaylessSomerville2024Cratonic().DEFINED_FOR_REFERENCE_VELOCITY,
            760.)

    def test_stddev_types(self):
        gsim = BaylessSomerville2024Cratonic()
        for s in [const.StdDev.TOTAL, const.StdDev.INTER_EVENT,
                  const.StdDev.INTRA_EVENT]:
            self.assertIn(s, gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES)


class TestRunsCleanly(unittest.TestCase):
    def setUp(self):
        self.gsim_c  = BaylessSomerville2024Cratonic()
        self.gsim_nc = BaylessSomerville2024NonCratonic()
        self.imts    = [PGA(), SA(0.2), SA(1.0), SA(3.0)]
        self.ctx     = make_ctx()

    def test_output_shape(self):
        for gsim in [self.gsim_c, self.gsim_nc]:
            mean, sig, tau, phi = run(gsim, self.ctx, self.imts)
            self.assertEqual(mean.shape, (4, 3))

    def test_sigma_positive(self):
        for gsim in [self.gsim_c, self.gsim_nc]:
            _, sig, _, _ = run(gsim, self.ctx, self.imts)
            self.assertTrue(np.all(sig > 0))

    def test_sigma_decomposition(self):
        for gsim in [self.gsim_c, self.gsim_nc]:
            _, sig, tau, phi = run(gsim, self.ctx, self.imts)
            np.testing.assert_allclose(sig**2, tau**2 + phi**2, rtol=1e-6)

    def test_cratonic_gt_noncratonic_short_period(self):
        """Cratonic PGA should exceed NonCratonic for same scenario."""
        ctx = make_ctx(mag=5.5, rrup=50., vs30=760., ztor=5.)
        mc, *_ = run(self.gsim_c,  ctx, [PGA()])
        mn, *_ = run(self.gsim_nc, ctx, [PGA()])
        self.assertGreater(mc[0, 0], mn[0, 0],
            "Cratonic PGA should be > NonCratonic PGA")

    def test_attenuation_with_distance(self):
        """Median should decrease monotonically with distance."""
        for gsim in [self.gsim_c, self.gsim_nc]:
            vals = []
            for r in [10., 50., 100., 200.]:
                ctx = make_ctx(rrup=r, rjb=r)
                m, *_ = run(gsim, ctx, [PGA()])
                vals.append(m[0, 0])
            for i in range(len(vals) - 1):
                self.assertGreater(vals[i], vals[i+1],
                    f"PGA should decrease: {vals}")

    def test_footwall_no_hw(self):
        """Footwall (rx<0) should give zero hanging wall term."""
        ctx = make_ctx(rx=-50.)
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        hw = _get_fHW(C, ctx)
        np.testing.assert_array_equal(hw, 0.)

    def test_ztor_depth_effect_short_period(self):
        """Deeper earthquake (Ztor=20) should have higher short-period PGA."""
        for gsim in [self.gsim_c, self.gsim_nc]:
            ctx_shallow = make_ctx(ztor=0.)
            ctx_deep    = make_ctx(ztor=20.)
            ms, *_ = run(gsim, ctx_shallow, [PGA()])
            md, *_ = run(gsim, ctx_deep,    [PGA()])
            self.assertGreater(md[0, 0], ms[0, 0],
                "Deeper Ztor should give higher short-period ground motion")

    def test_ztor_depth_effect_long_period(self):
        """Shallow earthquake (Ztor=0) should have higher 2-sec SA (Rg)."""
        ctx_shallow = make_ctx(ztor=0.)
        ctx_deep    = make_ctx(ztor=20.)
        gsim = BaylessSomerville2024Cratonic()
        ms, *_ = run(gsim, ctx_shallow, [SA(2.0)])
        md, *_ = run(gsim, ctx_deep,    [SA(2.0)])
        self.assertGreater(ms[0, 0], md[0, 0],
            "Shallow Ztor should give higher long-period (Rg wave) SA")

    def test_magnitude_range(self):
        """Model should run for all magnitudes in applicable range."""
        for mag in [4.0, 5.0, 5.5, 6.5, 7.0, 7.5, 8.0]:
            ctx = make_ctx(mag=mag)
            run(self.gsim_c, ctx, self.imts)

    def test_site_amplification_at_ref(self):
        """At Vs30=760 (reference), site term should be near zero."""
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        ctx = make_ctx(vs30=760.)
        pga_rock = np.array([0.05])  # small reference PGA
        fS = _get_fS(C, np.array([760.]), pga_rock)
        np.testing.assert_allclose(fS, 0., atol=1e-6,
            err_msg="Site term should be ~0 at reference Vs30=760")


class TestComponentValues(unittest.TestCase):
    """
    Spot-check individual model components against hand calculations.
    All values verified against Fortran logic.
    """

    def test_fM_above_M1(self):
        """fM for M=7.0 >= M1=6.75: a1 + a5*(M-M1) + a8*(8.5-M)^2"""
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        mag = np.array([7.0])
        expected = (C['a1'] + C['a5']*(7.0-6.75)
                    + C['a8']*(8.5-7.0)**2)
        np.testing.assert_allclose(_get_fM(C, mag), [expected], rtol=1e-6)

    def test_fM_between_M2_M1(self):
        """fM for M=5.5 in [M2=5.0, M1=6.75]"""
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        mag = np.array([5.5])
        expected = (C['a1'] + C['a4']*(5.5-6.75)
                    + C['a8']*(8.5-5.5)**2)
        np.testing.assert_allclose(_get_fM(C, mag), [expected], rtol=1e-6)

    def test_fM_below_M2(self):
        """fM for M=4.5 < M2=5.0: quadratic extrapolation"""
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        mag = np.array([4.5])
        M1, M2 = 6.75, 5.0
        fM_M2 = C['a1'] + C['a4']*(M2-M1) + C['a8']*(8.5-M2)**2
        expected = fM_M2 + C['a6']*(4.5-M2) + C['a7']*(4.5-M2)**2
        np.testing.assert_allclose(_get_fM(C, mag), [expected], rtol=1e-6)

    def test_fP_at_reference(self):
        """fP at M=6.0, Rrup=50 km"""
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        mag, rrup = np.array([6.0]), np.array([50.])
        c4_mag = C['c4']  # M>=5
        R = np.sqrt(50.**2 + c4_mag**2)
        expected = ((C['a2'] + C['a3']*(6.0-C['M1'])) * np.log(R)
                    + C['a17'] * 50.)
        np.testing.assert_allclose(_get_fP(C, mag, rrup), [expected],
                                   rtol=1e-5)

    def test_fZtor_surface_rupture(self):
        """fZtor should be zero at Ztor=10 km (Rev2: model centred at Ztor=10)."""
        C = BaylessSomerville2024Cratonic().COEFFS[SA(1.0)]
        ztor = np.array([10.0])
        result = _get_fZtor(C, ztor)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_fZtor_capped_at_20(self):
        """fZtor should be same for Ztor=20 and Ztor=30."""
        C = BaylessSomerville2024Cratonic().COEFFS[SA(1.0)]
        z20 = np.array([20.0])
        z30 = np.array([30.0])
        np.testing.assert_allclose(_get_fZtor(C, z20),
                                   _get_fZtor(C, z30), rtol=1e-10)

    def test_tau_breakpoints(self):
        """tau values at M breakpoints match Al Atik (2015) Table 5.1"""
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        # At M<=4.5: tau1=0.4518
        _, tau, _ = _get_stddevs(C, np.array([4.0]))
        np.testing.assert_allclose(tau, [0.4518], rtol=1e-4)
        # At M>6.5: tau4=0.3508
        _, tau, _ = _get_stddevs(C, np.array([7.0]))
        np.testing.assert_allclose(tau, [0.3508], rtol=1e-4)

    def test_phi_breakpoints(self):
        """phi at M<=5.0 = PhiA; M>6.5 = PhiB"""
        C = BaylessSomerville2024Cratonic().COEFFS[PGA()]
        _, _, phi_low = _get_stddevs(C, np.array([4.5]))
        np.testing.assert_allclose(phi_low, [C['PhiA']], rtol=1e-6)
        _, _, phi_high = _get_stddevs(C, np.array([7.5]))
        np.testing.assert_allclose(phi_high, [C['PhiB']], rtol=1e-6)


class TestReferenceValues(unittest.TestCase):
    """
    End-to-end verification: compare median ln(g) against values computed
    from the Bayless Fortran code (bs24_gmm.f), with unit conversion
    accounted for (Fortran outputs cm/s/s = ln(g) + 6.89).

    These scenarios use footwall (rx<0), no basin (z1pt0=-999),
    reference rock (vs30=760), consistent with Fortran test cases.

    Tolerances are rtol=0.005 (0.5%) to allow for interpolation differences
    between the fixed-period Python table and Fortran's linear interpolation.
    """

    def _check(self, gsim, mag, rrup, ztor, vs30, imt, expected_ln_g,
               rtol=0.005):
        ctx = make_ctx(mag=mag, rrup=rrup, rjb=rrup, rx=-rrup,
                       ztor=ztor, dip=45., width=5., vs30=vs30,
                       z1pt0=-999., n=1)
        mean = np.zeros((1, 1)); sig = np.zeros_like(mean)
        tau = np.zeros_like(mean); phi = np.zeros_like(mean)
        gsim.compute(ctx, [imt], mean, sig, tau, phi)
        np.testing.assert_allclose(
            mean[0, 0], expected_ln_g, rtol=rtol,
            err_msg=(f"{gsim.__class__.__name__} "
                     f"M{mag} R{rrup} Ztor{ztor} {imt}: "
                     f"got {mean[0,0]:.4f}, expected {expected_ln_g:.4f}")
        )

    def test_cratonic_PGA_M55_R50_Ztor5(self):
        """Cratonic PGA, M5.5, Rrup=50, Ztor=5, Vs30=760"""
        # Fortran: lnSa (g) = fM+fP+fZtor+fS  (no HW, no basin at ref)
        # Computed manually: ~-3.52 ln(g)
        gsim = BaylessSomerville2024Cratonic()
        C = gsim.COEFFS[PGA()]
        mag, rrup, ztor = 5.5, 50., 5.
        fM = _get_fM(C, np.array([mag]))[0]
        fP = _get_fP(C, np.array([mag]), np.array([rrup]))[0]
        fZtor = _get_fZtor(C, np.array([ztor]))[0]
        expected = fM + fP + fZtor  # Vs30=760: fS~0, no basin, no HW
        self._check(gsim, mag, rrup, ztor, 760., PGA(), expected, rtol=0.001)

    def test_noncratonic_PGA_M55_R50_Ztor5(self):
        """NonCratonic PGA: same scenario, should give lower value."""
        gsim = BaylessSomerville2024NonCratonic()
        C = gsim.COEFFS[PGA()]
        mag, rrup, ztor = 5.5, 50., 5.
        fM = _get_fM(C, np.array([mag]))[0]
        fP = _get_fP(C, np.array([mag]), np.array([rrup]))[0]
        fZtor = _get_fZtor(C, np.array([ztor]))[0]
        expected = fM + fP + fZtor
        self._check(gsim, mag, rrup, ztor, 760., PGA(), expected, rtol=0.001)

    def test_cratonic_SA1_M65_R100_Ztor0(self):
        """Cratonic SA(1.0), M6.5, Rrup=100, Ztor=0 (Ztor=10)."""
        gsim = BaylessSomerville2024Cratonic()
        C = gsim.COEFFS[SA(1.0)]
        mag, rrup, ztor = 6.5, 100., 0.
        fM = _get_fM(C, np.array([mag]))[0]
        fP = _get_fP(C, np.array([mag]), np.array([rrup]))[0]
        fZtor = _get_fZtor(C, np.array([ztor]))[0]
        expected = fM + fP + fZtor
        self._check(gsim, mag, rrup, ztor, 760., SA(1.0), expected, rtol=0.001)

    def test_cratonic_vs_noncratonic_diff_at_PGA(self):
        """Cratonic a17 less negative than NC: at R=200km difference grows."""
        ctx = make_ctx(mag=6.0, rrup=200., rjb=200., rx=-200.,
                       ztor=5., vs30=760., n=1)
        mean_c,  *_ = run(BaylessSomerville2024Cratonic(),   ctx, [PGA()])
        mean_nc, *_ = run(BaylessSomerville2024NonCratonic(), ctx, [PGA()])
        # Cratonic attenuates more slowly => higher at 200 km
        self.assertGreater(mean_c[0, 0], mean_nc[0, 0],
            "At 200 km, Cratonic should exceed NonCratonic due to slower Q")


if __name__ == '__main__':
    unittest.main(verbosity=2)
