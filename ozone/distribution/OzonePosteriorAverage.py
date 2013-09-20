"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from numpy.ma.core import asarray, mean
from ozone.distribution.OzonePosterior import OzonePosterior
import logging

class OzonePosteriorAverage(OzonePosterior):
    def __init__(self, num_estimates, prior):
        OzonePosterior.__init__(self, prior)
        
        self.num_estimates = num_estimates
        
    def log_likelihood(self, tau, kappa):
        logging.debug("Entering")
        logging.info("Computing %d likelihood estimates" % self.num_estimates)
        estimates = self.precompute_likelihood_estimates(tau, kappa)
        result = mean(estimates)
        logging.info("Average of %d likelihood estimates is %d" % 
                     (self.num_estimates, result))
        logging.debug("Leaving")
        return result
    
    def precompute_likelihood_estimates(self, tau, kappa):
        logging.debug("Entering")
        estimates = asarray([OzonePosterior.log_likelihood(self, tau, kappa) for _ in range(self.num_estimates)])
        logging.debug("Leaving")
        return estimates
