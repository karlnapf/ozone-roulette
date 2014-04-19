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

from engines.BatchClusterParameters import BatchClusterParameters
from engines.SGEComputationEngine import SGEComputationEngine
from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.output.StoreChainOutput import StoreChainOutput
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis
from numpy.lib.twodim_base import diag, eye
from numpy.ma.core import asarray
from os.path import expanduser
from ozone.distribution.OzonePosteriorAverageEngine import \
    OzonePosteriorAverageEngine
from pickle import dump
from tools.Log import Log
import logging
import os

def main():
    Log.set_loglevel(logging.DEBUG)
    
    prior = Gaussian(Sigma=eye(2) * 100)
    num_estimates = 5
    
    home = expanduser("~")
    folder = os.sep.join([home, "sample_ozone_posterior_average_sge"])
    
    # cluster admin set project jump for me to exclusively allocate nodes
    parameter_prefix = "#$ -P jump"
    
    cluster_parameters = BatchClusterParameters(foldername=folder,
                                            memory=4,
                                            loglevel=logging.DEBUG,
                                            parameter_prefix=parameter_prefix,
                                            max_walltime=36000)
        
    computation_engine = SGEComputationEngine(cluster_parameters, check_interval=10)
    posterior = OzonePosteriorAverageEngine(computation_engine=computation_engine,
                                        num_estimates=num_estimates,
                                        prior=prior)
    
    posterior.logdet_method = "shogun_estimate"
    
    proposal_cov = diag([ 4.000000000000000e-05, 1.072091680000000e+02])
    mcmc_sampler = StandardMetropolis(posterior, scale=1.0, cov=proposal_cov)
    
    start = asarray([-11.35, -13.1])
    mcmc_params = MCMCParams(start=start, num_iterations=2000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
#    chain.append_mcmc_output(PlottingOutput(None, plot_from=1, lag=1))
    chain.append_mcmc_output(StatisticsOutput(print_from=1, lag=1))
    
    store_chain_output = StoreChainOutput(folder, lag=50)
    chain.append_mcmc_output(store_chain_output)
    
    loaded = store_chain_output.load_last_stored_chain()
    if loaded is None:
        logging.info("Running chain from scratch")
    else:
        logging.info("Running chain from iteration %d" % loaded.iteration)
        chain = loaded
        
    chain.run()
    
    f = open(folder + os.sep + "final_chain", "w")
    dump(chain, f)
    f.close()

if __name__ == "__main__":
    main()
