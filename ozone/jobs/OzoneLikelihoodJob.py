
from ozone.jobs.OzoneJob import OzoneJob
from results.ScalarResult import ScalarResult

class OzoneLikelihoodJob(OzoneJob):
    def __init__(self, aggregator, ozone_posterior, tau, kappa):
        OzoneJob.__init__(self, aggregator, ozone_posterior, tau, kappa)
        
    def compute(self):
        result = self.ozone_posterior.log_likelihood(self.tau, self.kappa)
        result = ScalarResult(result)
        self.aggregator.submit_result(result)
