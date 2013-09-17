"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from aggregators.ScalarResultAggregator import ScalarResultAggregator
from numpy.ma.core import asarray
from ozone.distribution.OzonePosteriorRR import OzonePosteriorRR
from ozone.jobs.OzoneLikelihoodJob import OzoneLikelihoodJob

class OzonePosteriorRRCluster(OzonePosteriorRR):
    def __init__(self, computation_engine, rr_instance, num_estimates, prior):
        OzonePosteriorRR.__init__(self, rr_instance, num_estimates, prior)
        
        self.computation_engine = computation_engine
        
    def precompute_likelihood_estimates(self, tau, kappa):
        aggregators = []
        for _ in range(self.num_estimates):
            job = OzoneLikelihoodJob(ScalarResultAggregator(), self, tau, kappa)
            aggregators.append(self.computation_engine.submit_job(job))
        
        self.computation_engine.wait_for_all()
        
        results = []
        for i in range(self.num_estimates):
            aggregators[i].finalize()
            results.append(aggregators[i].get_final_result().result)
            
        return asarray(results)
