
from numpy.ma.core import var, mean
from ozone.distribution.OzonePosteriorAverageEngine import \
    OzonePosteriorAverageEngine
import logging

class OzonePosteriorRREngine(OzonePosteriorAverageEngine):
    def __init__(self, rr_instance, computation_engine, num_estimates, prior):
        OzonePosteriorAverageEngine.__init__(self, computation_engine, num_estimates, prior)
        
        self.rr_instance = rr_instance
        
    def log_likelihood(self, tau, kappa):
        estimates = self.precompute_likelihood_estimates(tau, kappa)
        
        if var(estimates) > 0:
            logging.info("Performing exponential Russian Roulette on %d precomputed samples" % 
                         len(estimates))
            rr_ified = self.rr_instance.exponential(estimates)
            return rr_ified
        else:
            logging.warn("Russian Roulette on one estimate not possible. Returning the estimate")
            return mean(estimates)
