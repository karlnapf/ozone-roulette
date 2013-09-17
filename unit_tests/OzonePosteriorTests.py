"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.ma.core import log
from numpy.random import randn
from ozone.distribution.OzonePosterior import OzonePosterior
from scipy.sparse.construct import spdiags
import unittest

class OzonePosteriorTests(unittest.TestCase):

    def test_log_likelihood_scikits_exact(self):
        tau = 2 ** (-11.35)
        kappa = 2 ** (-13.1)
        
        o = OzonePosterior(prior=None, logdet_method="scikits",
                           solve_method="scikits")
        
        self.assertAlmostEqual(o.log_likelihood(tau, kappa),
                               - 9.336375798558606e+05,
                               delta=0.001)

    def test_log_likelihood_shogun_exact(self):
        tau = 2 ** (-11.35)
        kappa = 2 ** (-13.1)
        
        o = OzonePosterior(prior=None, logdet_method="shogun_exact",
                           solve_method="shogun")
        
        self.assertAlmostEqual(o.log_likelihood(tau, kappa),
                               - 9.336375798558606e+05,
                               delta=0.5)
        
    def test_log_det_ozone_scikits_exact(self):
        o = OzonePosterior()
        kappa = 2 ** (-13.1)
        Q = o.create_Q_matrix(kappa)
        self.assertAlmostEqual(OzonePosterior.log_det_scikits(Q),
                               2.317769370813052e+06,
                               delta=0.001)
     
    def test_log_det_ozone_shogun_exact(self):
        o = OzonePosterior()
        kappa = 2 ** (-13.1)
        Q = o.create_Q_matrix(kappa)
        self.assertAlmostEqual(OzonePosterior.log_det_shogun_exact(Q),
                               2.317769370813052e+06,
                               delta=1)
    
    def test_load_data(self):
        OzonePosterior()
        
    def test_log_det_exact_toy_small_scikits(self):
        n = 3
        d = abs(randn(n))
        Q = spdiags(d, 0, n, n)
                    
        self.assertAlmostEqual(OzonePosterior.log_det_scikits(Q), sum(log(d)),
                               delta=1e-15)
        
    def test_log_det_exact_toy_small_shogun(self):
        n = 3
        d = abs(randn(n))
        Q = spdiags(d, 0, n, n)
                    
        self.assertAlmostEqual(OzonePosterior.log_det_shogun_exact(Q), sum(log(d)),
                               delta=1e-15)
        
    def test_log_det_exact_toy_large_scikits(self):
        n = 1e6
        d = abs(randn(n))
        Q = spdiags(d, 0, n, n)
                    
        self.assertAlmostEqual(OzonePosterior.log_det_scikits(Q), sum(log(d)),
                               delta=1e-5)
        
    def test_log_det_exact_toy_large_shogun(self):
        n = 1e6
        d = abs(randn(n))
        Q = spdiags(d, 0, n, n)
                    
        self.assertAlmostEqual(OzonePosterior.log_det_shogun_exact(Q), sum(log(d)),
                               delta=1e-5)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
