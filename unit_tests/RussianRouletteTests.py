
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
