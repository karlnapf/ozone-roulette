"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from ozone.jobs.OzoneJob import OzoneJob
from results.ScalarResult import ScalarResult
import logging

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
