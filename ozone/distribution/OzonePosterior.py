"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
"""
from main.distribution.Distribution import Distribution
from modshogun import CGMShiftedFamilySolver, DirectSparseLinearSolver, \
    LanczosEigenSolver, LogDetEstimator, LogRationalApproximationCGM, ProbingSampler, \
    RealSparseMatrixOperator, RealSparseMatrixOperator, SerialComputationEngine, \
    Statistics
from numpy.ma.core import shape, log, mean
from os.path import expanduser
from scipy.constants.constants import pi
from scipy.io.matlab.mio import loadmat
from scipy.sparse.construct import eye
from scipy.sparse.csc import csc_matrix
from tools.Log import Log
import os
# from scikits.sparse.cholmod import cholesky


class OzonePosterior(Distribution):
    def __init__(self, prior=None, logdet_method="shogun_exact",
                 solve_method="shogun"):
        Distribution.__init__(self, dimension=2)
        
        self.prior = prior
        self.logdet_method = logdet_method
        self.solve_method = solve_method
        
    def set_log_det_method(self, logdet_method):
        self.logdet_method = logdet_method    
        
    def set_solve_method(self, solve_method):
        self.solve_method = solve_method    
    
    @staticmethod
    def log_det_shogun_exact(Q):
        Log.debug("Entering")
        logdet = Statistics.log_det(csc_matrix(Q))
        Log.debug("Leaving")
        return logdet
    
    @staticmethod
    def log_det_scikits(Q):
#        d = cholesky(csc_matrix(Q)).L().diagonal()
#        return 2 * sum(log(d))
        raise Exception("cholmod not installed")
    
    @staticmethod
    def solve_sparse_linear_system_shogun(A, b):
        Log.debug("Entering")
        solver = DirectSparseLinearSolver()
        operator = RealSparseMatrixOperator(csc_matrix(A))
        result = solver.solve(operator, b)
        Log.debug("Leaving")
        return result
    
    @staticmethod
    def solve_sparse_linear_system_scikits(A, b):
#        factor = cholesky(A)
#        result = factor.solve_A(b)
#        return result
        raise Exception("cholmod not installed")
    
    @staticmethod
    def log_det_estimate_shogun(Q):
        Log.debug("Entering")
        op = RealSparseMatrixOperator(csc_matrix(Q))
        engine = SerialComputationEngine()
        linear_solver = CGMShiftedFamilySolver()
        accuracy = 1e-5
        eigen_solver = LanczosEigenSolver(op)
        eigen_solver.set_min_eigenvalue(1e-10)
        eigen_solver.compute()
        op_func = LogRationalApproximationCGM(op, engine, eigen_solver, linear_solver, accuracy)
        
        trace_sampler = ProbingSampler(op)
        log_det_estimator = LogDetEstimator(trace_sampler, op_func, engine)
        n_estimates = 1
        estimates = log_det_estimator.sample(n_estimates)
        
        Log.debug("Leaving")
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
        return Q + eye(Q.shape[0], Q.shape[1]) * 1e-10
    
    def log_det_method(self, Q):
        if self.logdet_method == "scikits":
            return OzonePosterior.log_det_scikits(Q)
        elif self.logdet_method == "shogun_estimate":
            return OzonePosterior.log_det_estimate_shogun(Q)
        elif self.logdet_method == "shogun_exact":
            return OzonePosterior.log_det_shogun_exact(Q)
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
        Log.debug("Entering")
        assert(shape(X)[0] == 1)
        result = self.log_likelihood(2 ** X[0, 0], 2 ** X[0, 1])
        
        if self.prior is not None:
            result += self.prior.log_pdf(X)
        
        Log.debug("Leaving")
        return  result
               
    def log_likelihood(self, tau, kappa):
        Log.debug("Entering")
        y, A = OzonePosterior.load_ozone_data()
        AtA = A.T.dot(A)
        
        Q = self.create_Q_matrix(kappa);
        n = len(y);
        M = Q + tau * AtA;
        
        Log.info("Computing log-determinants")
        logdet1 = self.log_det_method(Q)
        logdet2 = self.log_det_method(M)
        
        first = 0.5 * logdet1 + 0.5 * n * log(tau) - 0.5 * logdet2
        
        second_a = -0.5 * tau * (y.T.dot(y))
        
        Log.info("Computing rest of likelihood")
        second_b = A.T.dot(y)
        second_b = self.solve_sparse_linear_system(M, second_b)
        second_b = A.dot(second_b)
        second_b = y.T.dot(second_b)
        second_b = 0.5 * (tau ** 2) * second_b
        
        log_det_part = first
        quadratic_part = second_a + second_b
        const_part = -0.5 * n * log(2 * pi)
        
        log_marignal_lik = const_part + log_det_part + quadratic_part
        
        Log.debug("Leaving")
        return log_marignal_lik
