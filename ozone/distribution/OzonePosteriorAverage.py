
from numpy.ma.core import asarray, mean, var
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
        std_dev = std(estimates)
        logging.info("Average of %d likelihood estimates is %d +- %f" % 
                     (self.num_estimates, result, std_dev))
        logging.debug("Leaving")
        return result
    
    def precompute_likelihood_estimates(self, tau, kappa):
        logging.debug("Entering")
        estimates = asarray([OzonePosterior.log_likelihood(self, tau, kappa) for _ in range(self.num_estimates)])
        logging.debug("Leaving")
        return estimates
