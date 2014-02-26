"""
Copyright (c) 2013-2014 Heiko Strathmann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 *
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 *
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the author.
"""
from aggregators.ScalarResultAggregator import ScalarResultAggregator
from numpy.ma.core import log, zeros
from ozone.distribution.OzonePosterior import OzonePosterior
from ozone.distribution.OzonePosteriorAverage import OzonePosteriorAverage
from ozone.jobs.OzoneLikelihoodWithoutLogDetJob import \
    OzoneLikelihoodWithoutLogDetJob
from ozone.jobs.OzoneLogDetJob import OzoneLogDetJob
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
            aggregators_M.append(self.computation_engine.submit_job(job))
        
        # submit job for remainder of likelihood
        job = OzoneLikelihoodWithoutLogDetJob(ScalarResultAggregator(), self, tau, kappa)
        aggregator_remainder = self.computation_engine.submit_job(job)
        
        # grab a coffee
        self.computation_engine.wait_for_all()
        
        # collect results from all aggregators
        log_dets_Q = zeros(self.num_estimates)
        log_dets_M = zeros(self.num_estimates)
        for i in range(self.num_estimates):
            aggregators_Q[i].finalize()
            aggregators_M[i].finalize()
            log_dets_Q[i]=aggregators_Q[i].get_final_result().result
            log_dets_M[i]=aggregators_M[i].get_final_result().result
            aggregators_Q[i].clean_up()
            aggregators_M[i].clean_up()
            
        aggregator_remainder.finalize()
        result_remainder = aggregator_remainder.get_final_result().result
        aggregator_remainder.clean_up()
            
        # load n since needed for likelihood
        y, _ = OzonePosterior.load_ozone_data()
        n = len(y)
        
        # construct all likelihood estimates
        log_det_parts = 0.5 * log_dets_Q + 0.5 * n * log(tau) - 0.5 * log_dets_M
        estimates = log_det_parts + result_remainder
        
        logging.debug("Leaving")
        return estimates
