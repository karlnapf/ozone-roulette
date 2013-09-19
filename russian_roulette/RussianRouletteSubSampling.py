"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.ma.core import mean, zeros
from numpy.random import permutation
from russian_roulette.RussianRoulette import RussianRoulette
import logging

class RussianRouletteSubSampling(RussianRoulette):
    def __init__(self, threshold, block_size, num_desired_estimates=None):
        RussianRoulette.__init__(self, threshold)
        
        self.block_size = block_size
        self.num_desired_estimates = num_desired_estimates
        
    def exponential(self, estimates):
        subsampled = self.subsample(estimates)
        return RussianRoulette.exponential(self, subsampled)
    
    def subsample(self, estimates):
        logging.info("Sub-sampling %d estimates to get %d group estimate of size %d" %
                     (len(estimates), self.num_desired_estimates, self.block_size))
        assert(len(estimates) >= self.block_size)
        
        if self.num_desired_estimates is not None:
            subsampled = zeros(self.num_desired_estimates)
            
            for i in range(len(subsampled)):
                indices = permutation(len(estimates))[:self.block_size]
                subsampled[i] = mean(estimates[indices])
        else:
            num_samples = int(len(estimates) / self.block_size)
            subsampled = zeros(num_samples)
            for i in range(num_samples):
                idx_from = self.block_size * i
                subsampled[i] = mean(estimates[idx_from:idx_from + self.block_size])

        return subsampled
