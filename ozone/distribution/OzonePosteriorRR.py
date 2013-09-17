"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.ma.core import asarray, var, mean
from ozone.distribution.OzonePosterior import OzonePosterior

class OzonePosteriorRR(OzonePosterior):
    def __init__(self, rr_instance, num_estimates, prior):
        OzonePosterior.__init__(self, prior)
        
        self.rr_instance = rr_instance
        self.num_estimates = num_estimates
        
    def log_likelihood(self, tau, kappa):
        estimates = self.precompute_likelihood_estimates(tau, kappa)
        if var(estimates) > 0:
            rr_ified = self.rr_instance.exponential(estimates)
            return rr_ified
        else:
            return mean(estimates)
    
    def precompute_likelihood_estimates(self, tau, kappa):
        estimates = asarray([OzonePosterior.log_likelihood(self, tau, kappa) for _ in range(self.num_estimates)])
        return estimates
