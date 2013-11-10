"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from abc import abstractmethod
from jobs.IndependentJob import IndependentJob
from ozone.distribution.OzonePosterior import OzonePosterior

class OzoneJob(IndependentJob):
    def __init__(self, aggregator, ozone_posterior, tau, kappa):
        IndependentJob.__init__(self, aggregator)
        
        self.ozone_posterior = OzonePosterior(ozone_posterior.prior,
                                              ozone_posterior.logdet_method,
                                              ozone_posterior.solve_method)
        
        self.tau = tau
        self.kappa = kappa
    
    @abstractmethod
    def compute(self):
        raise NotImplementedError()
