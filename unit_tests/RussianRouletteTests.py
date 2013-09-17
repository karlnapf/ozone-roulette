"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from matplotlib.pyplot import plot, legend, show, clf
from numpy.linalg.linalg import norm
from numpy.ma.core import mean, zeros, arange, cumsum, ones, exp
from numpy.random import randn
from russian_roulette.RussianRoulette import RussianRoulette
from russian_roulette.RussianRouletteSubSampling import \
    RussianRouletteSubSampling
import unittest

class RussianRouletteTests(unittest.TestCase):

    def test_subsampling_resample(self):
        estimates = randn(1000)
        rr = RussianRouletteSubSampling(None, block_size=100, num_desired_estimates=1000)
        subsampled = rr.subsample(estimates)
        self.assertAlmostEqual(mean(estimates), mean(subsampled), delta=1e-2)
        
    def test_subsampling_no_resample(self):
        n = 1000
        block_size = 10
        estimates = randn(1000)
        rr = RussianRouletteSubSampling(None, block_size=block_size)
        subsampled = rr.subsample(estimates)
        self.assertEqual(n / block_size, len(subsampled))
        self.assertAlmostEqual(mean(estimates), mean(subsampled), delta=1e-2)

    def test_subsampling_no_resample_2(self):
        n = 1000
        block_size = 1
        estimates = randn(1000)
        rr = RussianRouletteSubSampling(None, block_size=block_size)
        subsampled = rr.subsample(estimates)
        self.assertEqual(n / block_size, len(subsampled))
        self.assertAlmostEqual(norm(estimates - subsampled), 0)
                               
    def test_rr_exp_classic(self):
        rr = RussianRoulette(1e-5)
        n = 500
        
        log_expected_value = 1
        std_dev = 10
        log_estimator = lambda n: randn(n) * std_dev + log_expected_value
        
        est_biased = lambda : exp(mean(log_estimator(n)))
        est_rr = lambda : exp(rr.exponential(log_estimator(n)))
        
        m = 1000
        samples_biased = zeros(m)
        samples_rr = zeros(m)
        
        for i in range(m):
            samples_biased[i] = est_biased()
            samples_rr[i] = est_rr()
            
        normaliser = arange(m) + 1
        plot(normaliser, cumsum(samples_biased) / normaliser)
        plot(normaliser, cumsum(samples_rr) / normaliser)
        plot([1, m], exp(ones(2) * log_expected_value))
        legend(["Biased", "Russian Roulette", "True"])
        show()

    def test_rr_exp_subsampling(self):
        n = 200
        rr = RussianRouletteSubSampling(threshold=1e-2, block_size=100, num_desired_estimates=n)
        
        log_expected_value = 1
        std_dev = 10
        log_estimator = lambda n: randn(n) * std_dev + log_expected_value
        
        est_biased = lambda : exp(mean(log_estimator(n)))
        est_rr = lambda : exp(rr.exponential(log_estimator(n)))
        
        m = 500
        samples_biased = zeros(m)
        samples_rr = zeros(m)
        
        for i in range(m):
            samples_biased[i] = est_biased()
            samples_rr[i] = est_rr()
            
        print mean(samples_biased)
        print mean(samples_rr)
            
        normaliser = arange(m) + 1
        clf()
        plot(normaliser, cumsum(samples_biased) / normaliser)
        plot(normaliser, cumsum(samples_rr) / normaliser)
        plot([1, m], exp(ones(2) * log_expected_value))
        legend(["Biased", "Russian Roulette", "True"])
        show()

    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
