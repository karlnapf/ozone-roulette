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
import logging
from matplotlib.pyplot import plot, legend, show, clf
from numpy.linalg.linalg import norm
from numpy.ma.core import mean, zeros, arange, cumsum, ones, exp
from numpy.random import randn
import unittest

from independent_jobs.tools.Log import Log
from russian_roulette.RussianRoulette import RussianRoulette


class RussianRouletteTests(unittest.TestCase):
    def test_equal_estimates(self):
        Log.set_loglevel(logging.DEBUG)
        rr = RussianRoulette(1e-5, block_size=100)
        
        log_estimates=randn(1000)
        log_estimates=ones(1000)*(-942478.011941)
        print rr.exponential(log_estimates)

#     def test_rr_exp_classic(self):
#         rr = RussianRoulette(1e-5)
#         n = 500
#          
#         log_expected_value = 1
#         std_dev = 10
#         log_estimator = lambda n: randn(n) * std_dev + log_expected_value
#          
#         est_biased = lambda : exp(mean(log_estimator(n)))
#         est_rr = lambda : exp(rr.exponential(log_estimator(n)))
#          
#         m = 1000
#         samples_biased = zeros(m)
#         samples_rr = zeros(m)
#          
#         for i in range(m):
#             samples_biased[i] = est_biased()
#             samples_rr[i] = est_rr()
#              
#         normaliser = arange(m) + 1
#         plot(normaliser, cumsum(samples_biased) / normaliser)
#         plot(normaliser, cumsum(samples_rr) / normaliser)
#         plot([1, m], exp(ones(2) * log_expected_value))
#         legend(["Biased", "Russian Roulette", "True"])
#         show()

    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
