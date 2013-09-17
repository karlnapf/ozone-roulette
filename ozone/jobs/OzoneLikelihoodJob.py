"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from jobs.IndependentJob import IndependentJob
from ozone.distribution.OzonePosterior import OzonePosterior
from results.ScalarResult import ScalarResult

class OzoneLikelihoodJob(IndependentJob):
    def __init__(self, aggregator, ozone_posterior, tau, kappa):
        IndependentJob.__init__(self, aggregator)
        
        self.ozone_posterior = OzonePosterior(ozone_posterior.prior)
        self.tau = tau
        self.kappa = kappa
    
    def compute(self):
        result = self.ozone_posterior.log_likelihood(self.tau, self.kappa)
        result = ScalarResult(result)
        self.aggregator.submit_result(result)
