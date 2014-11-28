
import logging

from independent_jobs.results.ScalarResult import ScalarResult
from ozone.jobs.OzoneJob import OzoneJob


class OzoneLikelihoodWithoutLogDetJob(OzoneJob):
    def __init__(self, aggregator, ozone_posterior, tau, kappa):
        OzoneJob.__init__(self, aggregator, ozone_posterior, tau, kappa)
    
    def compute(self):
        logging.debug("Entering")
        
        lik_wihtout_logdet = \
        self.ozone_posterior.log_likelihood_without_logdet(self.tau, self.kappa)
                
        result = ScalarResult(lik_wihtout_logdet)
        self.aggregator.submit_result(result)
        
        logging.debug("Leaving")
