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
