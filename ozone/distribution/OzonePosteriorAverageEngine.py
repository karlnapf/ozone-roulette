
import logging
from numpy import where
from numpy.ma.core import log, zeros

from independent_jobs.aggregators.ScalarResultAggregator import ScalarResultAggregator
from ozone.distribution.OzonePosterior import OzonePosterior
from ozone.distribution.OzonePosteriorAverage import OzonePosteriorAverage
from ozone.jobs.OzoneLikelihoodWithoutLogDetJob import \
    OzoneLikelihoodWithoutLogDetJob
from ozone.jobs.OzoneLogDetJob import OzoneLogDetJob


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
            log_dets_Q[i] = aggregators_Q[i].get_final_result().result
            log_dets_M[i] = aggregators_M[i].get_final_result().result
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
        
        # crude check for an overflow to print error details
        limit = 1e100
        indices = where(abs(estimates) > limit)[0]
        if len(indices) > 0:
            logging.info("Log-likelihood estimates overflow occured at the following indices:")
            for idx in indices:
                logging.info("At index %d. Details are: " % idx)
                logging.info("log-det Q: " + aggregators_Q[idx].job_name + 
                             ". Result is %f" % log_dets_Q[idx])
                logging.info("log-det M: " + aggregators_M[idx].job_name + 
                             ". Result is %f" % log_dets_M[idx])
                logging.info("log-lik-without-log-det: " + 
                             aggregator_remainder.job_name + ". Result is %f" % result_remainder[idx])
                
            logging.info("Removing mentioned estimates from list")
            estimates = estimates[abs(estimates) < limit]
            logging.info("New number of estimates is %d, old was %d" % 
                         (len(estimates), self.num_estimates))
                
        
        logging.debug("Leaving")
        return estimates
