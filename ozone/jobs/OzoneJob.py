
from abc import abstractmethod

from independent_jobs.jobs.IndependentJob import IndependentJob
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
