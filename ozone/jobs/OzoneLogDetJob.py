
import logging

from independent_jobs.results.ScalarResult import ScalarResult
from ozone.distribution.OzonePosterior import OzonePosterior
from ozone.jobs.OzoneJob import OzoneJob


class OzoneLogDetJob(OzoneJob):
    def __init__(self, aggregator, ozone_posterior, tau, kappa, matrix_type):
        OzoneJob.__init__(self, aggregator, ozone_posterior, tau, kappa)
        
        self.matrix_type = matrix_type
    
    def compute(self):
        logging.debug("Entering")
        
        # needed for both matrices
        Q = self.ozone_posterior.create_Q_matrix(self.kappa);
        
        if self.matrix_type == "Q":
            logging.info("Matrix type Q")
            logdet = self.ozone_posterior.log_det_method(Q)
        elif self.matrix_type == "M":
            logging.info("Matrix type M")
            _, A = OzonePosterior.load_ozone_data()
            AtA = A.T.dot(A)
            M = Q + self.tau * AtA;
            logdet = self.ozone_posterior.log_det_method(M)
        else:
            raise ValueError("Unknown matrix type: %s" % self.matrix_type)
        
        result = ScalarResult(logdet)
        self.aggregator.submit_result(result)
        
        logging.debug("Leaving")
