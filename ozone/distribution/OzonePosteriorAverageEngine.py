"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from aggregators.ScalarResultAggregator import ScalarResultAggregator
from numpy.ma.core import log, zeros
from ozone.distribution.OzonePosterior import OzonePosterior
from ozone.distribution.OzonePosteriorAverage import OzonePosteriorAverage
from ozone.jobs.OzoneLikelihoodWithoutLogDetJob import OzoneLogDetJob
from ozone.jobs.OzoneLogDetJob import OzoneLikelihoodWithoutLogDetJob
import logging

class OzonePosteriorAverageEngine(OzonePosteriorAverage):
    def __init__(self, computation_engine, num_estimates, prior):
        OzonePosteriorAverage.__init__(self, num_estimates, prior)
        
        self.computation_engine = computation_engine
        
    def precompute_likelihood_estimates(self, tau, kappa):
        logging.debug("Entering")
        
        # submit all jobs for log-determinant Q
        aggregators_Q = []
        for _ in range(self.num_estimates):
            job = OzoneLogDetJob(ScalarResultAggregator(), self, tau, kappa, "Q")
            aggregators_Q.append(self.computation_engine.submit_job(job))
        
        # submit all jobs for log-determinant M
        aggregators_M = []
        for _ in range(self.num_estimates):
            job = OzoneLogDetJob(ScalarResultAggregator(), self, tau, kappa, "M")
            aggregators_Q.append(self.computation_engine.submit_job(job))
        
        # submit job for remainder of likelihood
        job = OzoneLikelihoodWithoutLogDetJob(ScalarResultAggregator(), self, tau, kappa)
        aggregator_remainder = self.computation_engine.submit_job(job)
        
        # grab a coffee
        self.computation_engine.wait_for_all()
        
        # collect results from all aggregators
        log_dets_Q = zeros(self.num_estimates)
        for i in range(self.num_estimates):
            aggregators_Q[i].finalize()
            log_dets_Q[i]=aggregators_Q[i].get_final_result().result
            
        log_dets_M = zeros(self.num_estimates)
        for i in range(self.num_estimates):
            aggregators_M[i].finalize()
            log_dets_M[i]=aggregators_M[i].get_final_result().result
            
        aggregator_remainder.finalize()
        result_remainder = aggregator_remainder.get_final_result().result
            
        # load n since needed for likelihood
        y, _ = OzonePosterior.load_ozone_data()
        n = len(y)
        
        # construct all likelihood estimates
        log_det_parts = 0.5 * log_dets_Q + 0.5 * n * log(tau) - 0.5 * log_dets_M
        estimates = log_det_parts + result_remainder
        
        return estimates
        logging.debug("Leaving")
