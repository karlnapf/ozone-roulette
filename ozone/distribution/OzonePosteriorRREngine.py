"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.ma.core import var, mean
from ozone.distribution.OzonePosteriorAverageEngine import \
    OzonePosteriorAverageEngine
import logging

class OzonePosteriorRREngine(OzonePosteriorAverageEngine):
    def __init__(self, rr_instance, num_estimates, prior):
        OzonePosteriorAverageEngine.__init__(self, num_estimates, prior)
        
        self.rr_instance = rr_instance
        
    def log_likelihood(self, tau, kappa):
        estimates = self.precompute_likelihood_estimates(tau, kappa)
        
        if var(estimates) > 0:
            logging.info("Performing exponential Russian Roulette on %d precomputed samples" %
                         self.num_estimates)
            rr_ified = self.rr_instance.exponential(estimates)
            return rr_ified
        else:
            return mean(estimates)
