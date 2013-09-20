"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from abc import abstractmethod
from numpy.lib.function_base import delete
from numpy.ma.core import mean, zeros, log, arange, std
from numpy.random import permutation, rand
import logging

class RussianRoulette(object):
    def __init__(self, threshold, block_size=1):
        self.threshold = threshold
        self.block_size = block_size

    @abstractmethod
    def get_estimate(self, estimates, index):
        start_idx = index * self.block_size
        stop_idx = index * self.block_size + self.block_size
        
        # if there are enough samples, use them, sub-sample if not
        if stop_idx < len(estimates):
            logging.info("Averaging over %d samples from index %d to %d" % 
                         (self.block_size, start_idx, stop_idx))
            indices = arange(start_idx, stop_idx)
        else:
            logging.info("Averaging over a random subset of %d samples" % 
                         self.block_size)
            
            indices = permutation(len(estimates))[:self.block_size]
        
        return mean(estimates[indices])
            
    def exponential(self, estimates):
        logging.debug("Entering")
        
        # find a strict lower bound on the estimates and remove it from list
        bound = estimates.min()
        bound_idx = estimates.argmin()
        estimates = delete(estimates, bound_idx)
        estimates = estimates - bound
        

        # find an integer close to the mean of the transformed estimates and divide
        E = max(int(round(mean(estimates))), 1)
        estimates = estimates / E
        
        logging.info("Using %f as lower bound on estimates" % bound)
        logging.info("Computing product of E=%d RR estimates" % E)
        logging.info("Std-deviation after scaling is" % std(estimates))
        
        # index for iterating through the used estimates
        # (might be averaged, so might be lower than the number of available estimates
        # if the block size is greater than one
        estimate_idx = 0
        
        samples = zeros(E)
        for iteration in range(E):
            weight = 1
            
            # start with x^0 which is 1
            samples[iteration] = 1
            term = 1
            
            # index for computed samples
            series_term_idx = 1

            while weight > 0:
                # update current term of infinite series
                # average over block
                x_inner = self.get_estimate(estimates, estimate_idx)
                term *= (x_inner / series_term_idx)
                
                # if summation has reached threshold, update weights
                if abs(term) < self.threshold:
                    q = term / self.threshold
                    if rand() < q:
                        # continue and update weight
                        weight = weight / q
                    else:
                        # stop summation
                        weight = 0
            
                samples[iteration] += weight * term;
                estimate_idx += 1
                series_term_idx += 1
                
            logging.info("RR estimate %d/%d took %d series terms" % 
                         (iteration, E, series_term_idx))
        # now put things together. Note that samples contains an unbiased estimate
        # which might be quite small. However, due to the removal of the bound,
        # this will not cause an underflow and we can just take the log.
        logging.debug("Leaving")
        return bound + sum(log(samples));
