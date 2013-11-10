"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from abc import abstractmethod
from main.distribution.Distribution import Distribution
from modshogun import CGMShiftedFamilySolver, DirectSparseLinearSolver, \
    LanczosEigenSolver, LogDetEstimator, LogRationalApproximationCGM, ProbingSampler, \
    RealSparseMatrixOperator, RealSparseMatrixOperator, SerialComputationEngine, \
    Statistics
from numpy.ma.core import shape, log, mean
from numpy.random import randn
from os.path import expanduser
from scipy.constants.constants import pi
from scipy.io.matlab.mio import loadmat
from scipy.sparse.construct import eye
from scipy.sparse.csc import csc_matrix
import logging
import os
# from scikits.sparse.cholmod import cholesky

class OzonePosterior(Distribution):
    ridge = 1
    
    def __init__(self, prior=None, logdet_method="shogun_exact",
                 solve_method="shogun", shogun_loglevel=2):
        Distribution.__init__(self, dimension=2)
        
        self.prior = prior
        self.logdet_method = logdet_method
        self.solve_method = solve_method
        
        LogDetEstimator().io.set_loglevel(shogun_loglevel)
        LogDetEstimator().io.set_location_info(1)
        
    def set_log_det_method(self, logdet_method):
        self.logdet_method = logdet_method    
        
    def set_solve_method(self, solve_method):
        self.solve_method = solve_method    
    
    @staticmethod
    def log_det_shogun_exact(Q):
        logging.debug("Entering")
        logdet = Statistics.log_det(csc_matrix(Q))
        logging.debug("Leaving")
        return logdet
    
    @staticmethod
    def log_det_shogun_exact_plus_noise(Q):
        logging.debug("Entering")
        logdet = Statistics.log_det(csc_matrix(Q)) + randn()
        logging.debug("Leaving")
        return logdet
    
    @staticmethod
    def log_det_scikits(Q):
#        d = cholesky(csc_matrix(Q)).L().diagonal()
#        return 2 * sum(log(d))
        raise Exception("cholmod not installed")
    
    @staticmethod
    def solve_sparse_linear_system_shogun(A, b):
        logging.debug("Entering")
        solver = DirectSparseLinearSolver()
        operator = RealSparseMatrixOperator(csc_matrix(A))
        result = solver.solve(operator, b)
        logging.debug("Leaving")
        return result
    
    @staticmethod
    def solve_sparse_linear_system_scikits(A, b):
#        factor = cholesky(A)
#        result = factor.solve_A(b)
#        return result
        raise Exception("cholmod not installed")
    
    @staticmethod
    def log_det_estimate_shogun(Q):
        logging.debug("Entering")
        op = RealSparseMatrixOperator(csc_matrix(Q))
        engine = SerialComputationEngine()
        linear_solver = CGMShiftedFamilySolver()
        accuracy = 1e-3
        eigen_solver = LanczosEigenSolver(op)
        eigen_solver.set_min_eigenvalue(OzonePosterior.ridge)
        op_func = LogRationalApproximationCGM(op, engine, eigen_solver, linear_solver, accuracy)

        # limit computation time
        linear_solver.set_iteration_limit(1000)
        eigen_solver.set_max_iteration_limit(1000)
        
        logging.info("Computing Eigenvalues (only largest)")
        eigen_solver.compute()
        
        trace_sampler = ProbingSampler(op)
        log_det_estimator = LogDetEstimator(trace_sampler, op_func, engine)
        n_estimates = 1
        logging.info("Sampling log-determinant with probing vectors and rational approximation")
        estimates = log_det_estimator.sample(n_estimates)
        
        logging.debug("Leaving")
        return mean(estimates)
        
    @staticmethod
    def get_data_folder():
        home = expanduser("~")
        return os.sep.join([home, "data", "ozone"]) + os.sep
    
    def create_Q_matrix(self, kappa):
        folder = OzonePosterior.get_data_folder()
        
        GiCG = loadmat(folder + "GiCG.mat")["GiCG"]
        G = loadmat(folder + "G.mat")["G"]
        C0 = loadmat(folder + "C0.mat")["C0"]
        
        Q = GiCG + 2 * (kappa ** 2) * G + (kappa ** 4) * C0
        return Q + eye(Q.shape[0], Q.shape[1]) * OzonePosterior.ridge
    
    def log_det_method(self, Q):
        if self.logdet_method == "scikits":
            return OzonePosterior.log_det_scikits(Q)
        elif self.logdet_method == "shogun_estimate":
            return OzonePosterior.log_det_estimate_shogun(Q)
        elif self.logdet_method == "shogun_exact":
            return OzonePosterior.log_det_shogun_exact(Q)
        elif self.logdet_method == "shogun_exact_plus_noise":
            return OzonePosterior.log_det_shogun_exact_plus_noise(Q)
        else:
            raise ValueError("Log-det method unknown")
        
    def solve_sparse_linear_system(self, A, b):
        if self.solve_method == "scikits":
            return OzonePosterior.solve_sparse_linear_system_scikits(A, b)
        elif self.solve_method == "shogun":
            return OzonePosterior.solve_sparse_linear_system_shogun(A, b)
        else:
            raise ValueError("Solve method method unknown")
        
    @staticmethod
    def load_ozone_data():
        folder = OzonePosterior.get_data_folder()
        
        y = loadmat(folder + "y.mat")["y"][:, 0]
        assert(len(shape(y)) == 1)
        
        A = loadmat(folder + "A.mat")["A"]
        
        return y, A
            
    def log_pdf(self, X):
        logging.debug("Entering")
        assert(shape(X)[0] == 1)
        result = self.log_likelihood(2 ** X[0, 0], 2 ** X[0, 1])
        
        if self.prior is not None:
            result += self.prior.log_pdf(X)
        
        logging.debug("Leaving")
        return  result
               
    def log_likelihood(self, tau, kappa):
        logging.debug("Entering")
        log_det_part = self.log_likelihood_logdet(tau, kappa)
        other_part = self.log_likelihood_without_logdet(tau, kappa)
        
        log_marignal_lik = log_det_part + other_part
        
        logging.debug("Leaving")
        return log_marignal_lik
    
    @abstractmethod
    def log_likelihood_logdet(self, tau, kappa):
        logging.debug("Entering")
        y, A = OzonePosterior.load_ozone_data()
        AtA = A.T.dot(A)
        
        Q = self.create_Q_matrix(kappa)
        n = len(y)
        M = Q + tau * AtA
        
        logdet1 = self.log_det_method(Q)
        logdet2 = self.log_det_method(M)
        
        log_det_part = 0.5 * logdet1 + 0.5 * n * log(tau) - 0.5 * logdet2
        
        logging.debug("Leaving")
        return log_det_part
    
    def log_likelihood_without_logdet(self, tau, kappa):
        logging.debug("Entering")
        y, A = OzonePosterior.load_ozone_data()
        AtA = A.T.dot(A)
        
        Q = self.create_Q_matrix(kappa);
        n = len(y);
        M = Q + tau * AtA;
        
        second_a = -0.5 * tau * (y.T.dot(y))
        
        second_b = A.T.dot(y)
        second_b = self.solve_sparse_linear_system(M, second_b)
        second_b = A.dot(second_b)
        second_b = y.T.dot(second_b)
        second_b = 0.5 * (tau ** 2) * second_b
        
        quadratic_part = second_a + second_b
        const_part = -0.5 * n * log(2 * pi)
        
        result = const_part + quadratic_part
        
        logging.debug("Leaving")
        return result
