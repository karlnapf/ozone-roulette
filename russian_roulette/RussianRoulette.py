"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.lib.function_base import delete
from numpy.ma.core import mean, zeros, log
from numpy.random import rand

class RussianRoulette(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def exponential(self, estimates):
        # find a strict lower bound on the estimates and remove it from list
        bound = estimates.min()
        bound_idx = estimates.argmin()
        estimates = delete(estimates, bound_idx)
        estimates = estimates - bound

        # find an integer close to the mean of the transformed estimates and divide
        E = max(int(round(mean(estimates))), 1)
        estimates = estimates / E
        estimate_idx = 0
        
        samples = zeros(E)
        for iteration in range(E):
#            print "RR estimate %d/%d" % (iteration, E)
            weight = 1
            
            # start with x^0 which is 1
            samples[iteration] = 1
            term = 1
            
            # index for computed samples
            series_term_idx = 1

            while weight > 0:
                # update current term of infinite series
                x_inner = estimates[estimate_idx]
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
            
#                print "RR iteration %d, term=%f, thresh=%f, weight=%f" % \
#                    (iteration, term, self.threshold, weight)
            
                samples[iteration] += weight * term;
                estimate_idx += 1
                series_term_idx += 1
                
                
#        print "used %d/%d estimates" % (estimate_idx, len(estimates))
        # now put things together. Note that samples contains an unbiased estimate
        # which might be quite small. However, due to the removal of the bound,
        # this will not cause an underflow and we can just take the log.
        return bound + sum(log(samples));
