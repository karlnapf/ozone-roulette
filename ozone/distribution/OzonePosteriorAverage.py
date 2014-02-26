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
from numpy.ma.core import asarray, mean
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
        logging.info("Average of %d likelihood estimates is %d" % 
                     (self.num_estimates, result))
        logging.debug("Leaving")
        return result
    
    def precompute_likelihood_estimates(self, tau, kappa):
        logging.debug("Entering")
        estimates = asarray([OzonePosterior.log_likelihood(self, tau, kappa) for _ in range(self.num_estimates)])
        logging.debug("Leaving")
        return estimates
