"""
Copyright (c) 2013-2014 Heiko Strathmann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 *
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the author.
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
