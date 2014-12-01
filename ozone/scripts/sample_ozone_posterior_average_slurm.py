import logging
from numpy.lib.twodim_base import diag, eye
from numpy.ma.core import asarray
import os
from os.path import expanduser
from pickle import dump

from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.FileSystem import FileSystem
from independent_jobs.tools.Log import Log
from kameleon_mcmc.distribution.Gaussian import Gaussian
from kameleon_mcmc.mcmc.MCMCChain import MCMCChain
from kameleon_mcmc.mcmc.MCMCParams import MCMCParams
from kameleon_mcmc.mcmc.output.StatisticsOutput import StatisticsOutput
from kameleon_mcmc.mcmc.output.StoreChainOutput import StoreChainOutput
from kameleon_mcmc.mcmc.samplers.StandardMetropolis import StandardMetropolis
from ozone.distribution.OzonePosteriorAverageEngine import \
    OzonePosteriorAverageEngine


def main():
    Log.set_loglevel(logging.DEBUG)
    
    modulename = "sample_ozone_posterior_average_slurm"
    if not FileSystem.cmd_exists("sbatch"):
        engine = SerialComputationEngine()
    else:
        johns_slurm_hack = "#SBATCH --partition=intel-ivy,wrkstn,compute"
        folder = os.sep + os.sep.join(["nfs", "data3", "ucabhst", modulename])
        batch_parameters = BatchClusterParameters(foldername=folder, max_walltime=60 * 60,
                                                  resubmit_on_timeout=False, memory=3,
                                                  parameter_prefix=johns_slurm_hack)
        engine = SlurmComputationEngine(batch_parameters, check_interval=1,
                                do_clean_up=True)
    
    
    prior = Gaussian(Sigma=eye(2) * 100)
    num_estimates = 1
    
    posterior = OzonePosteriorAverageEngine(computation_engine=engine,
                                        num_estimates=num_estimates,
                                        prior=prior)
    posterior.logdet_method = "shogun_estimate"
    
    proposal_cov = diag([ 4.000000000000000e-05, 1.072091680000000e+02])
    mcmc_sampler = StandardMetropolis(posterior, scale=1.0, cov=proposal_cov)
    
    start = asarray([-11.35, -13.1])
    mcmc_params = MCMCParams(start=start, num_iterations=2000)
    chain = MCMCChain(mcmc_sampler, mcmc_params)
    
    chain.append_mcmc_output(StatisticsOutput(print_from=1, lag=1))
    
    home = expanduser("~")
    folder = os.sep.join([home, modulename])
    store_chain_output = StoreChainOutput(folder)
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
