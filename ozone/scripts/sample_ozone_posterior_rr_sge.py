

import logging
from numpy.lib.twodim_base import diag, eye
from numpy.ma.core import asarray
import os
from os.path import expanduser
from pickle import dump

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SGEComputationEngine import SGEComputationEngine
from independent_jobs.tools.Log import Log
from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.output.StoreChainOutput import StoreChainOutput
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis
from ozone.distribution.OzonePosteriorRREngine import OzonePosteriorRREngine
from russian_roulette.RussianRoulette import RussianRoulette


def main():
    Log.set_loglevel(logging.DEBUG)
    
    prior = Gaussian(Sigma=eye(2) * 100)
    num_estimates = 1000
    
    home = expanduser("~")
    folder = os.sep.join([home, "sample_ozone_posterior_rr_sge"])
    
    # cluster admin set project jump for me to exclusively allocate nodes
    parameter_prefix = ""  # #$ -P jump"
    
    cluster_parameters = BatchClusterParameters(foldername=folder,
                                            memory=7.8,
                                            loglevel=logging.DEBUG,
                                            parameter_prefix=parameter_prefix,
                                            max_walltime=60 * 60 * 24 - 1)
        
    computation_engine = SGEComputationEngine(cluster_parameters, check_interval=10)
    
    rr_instance = RussianRoulette(1e-3, block_size=400)
    
    posterior = OzonePosteriorRREngine(rr_instance=rr_instance,
                                       computation_engine=computation_engine,
                                       num_estimates=num_estimates,
                                       prior=prior)
    
    posterior.logdet_method = "shogun_estimate"
    
    proposal_cov = diag([ 4.000000000000000e-05, 1.072091680000000e+02])
    mcmc_sampler = StandardMetropolis(posterior, scale=1.0, cov=proposal_cov)
    
    start = asarray([-11.55, -10.1])
    mcmc_params = MCMCParams(start=start, num_iterations=5000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
#    chain.append_mcmc_output(PlottingOutput(None, plot_from=1, lag=1))
    chain.append_mcmc_output(StatisticsOutput(print_from=1, lag=1))
    
    store_chain_output = StoreChainOutput(folder, lag=1)
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
