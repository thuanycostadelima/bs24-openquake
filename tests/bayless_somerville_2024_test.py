# -*- coding: utf-8 -*-
"""
GEM-style verification tests for the Bayless and Somerville (2024) GMM.

Tests compare the Python implementation against pre-computed reference
values stored in the CSV files:
    BS24_CRATONIC.csv
    BS24_NONCRATONIC.csv

Run with:
    python bayless_somerville_2024_test.py
or via pytest when submitted to GEM.

Reference values were computed from the Python implementation itself,
verified against the Fortran reference subroutine (bs24_gmm.f) provided
by Jeff Bayless (jeff.bayless@aecom.com). Agreement with Fortran is to
within floating-point precision (max diff 0.000045 ln units on mean SA,
max diff 0.000005 ln units on sigma) across 1760 test cases.

Author: Thuany Costa de Lima, Geoscience Australia (May 2026)
"""
import os
import unittest

from openquake.hazardlib.gsim.bayless_somerville_2024 import (
    BaylessSomerville2024Cratonic,
    BaylessSomerville2024NonCratonic,
)
from openquake.hazardlib.tests.gsim.utils import BaseGSIMTestCase

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'BS24')


class BaylessSomerville2024CratonicTestCase(BaseGSIMTestCase):
    """
    Verification tests for the Cratonic version of BS24.
    Applicable to Yilgarn, Gawler, Pilbara, Kimberley, and Northern
    Australian Cratons (NSHA23 classification).
    """
    GSIM_CLASS = BaylessSomerville2024Cratonic

    def test_mean(self):
        self.check('BS24/BS24_CRATONIC.csv',
                   max_discrep_percentage=0.1)

    def test_std_total(self):
        self.check('BS24/BS24_CRATONIC.csv',
                   max_discrep_percentage=0.1)

    def test_std_inter(self):
        self.check('BS24/BS24_CRATONIC.csv',
                   max_discrep_percentage=0.1)

    def test_std_intra(self):
        self.check('BS24/BS24_CRATONIC.csv',
                   max_discrep_percentage=0.1)


class BaylessSomerville2024NonCratonicTestCase(BaseGSIMTestCase):
    """
    Verification tests for the NonCratonic version of BS24.
    Applicable to Eastern Australian Phanerozoic Accretionary Terranes,
    extended and oceanic crust (NSHA23 classification).
    """
    GSIM_CLASS = BaylessSomerville2024NonCratonic

    def test_mean(self):
        self.check('BS24/BS24_NONCRATONIC.csv',
                   max_discrep_percentage=0.1)

    def test_std_total(self):
        self.check('BS24/BS24_NONCRATONIC.csv',
                   max_discrep_percentage=0.1)

    def test_std_inter(self):
        self.check('BS24/BS24_NONCRATONIC.csv',
                   max_discrep_percentage=0.1)

    def test_std_intra(self):
        self.check('BS24/BS24_NONCRATONIC.csv',
                   max_discrep_percentage=0.1)


if __name__ == '__main__':
    unittest.main()
