
from jobs.DummyJob import DummyJob

class ShogunJobImportLogdet(DummyJob):
    def __init__(self, aggregator, sleep_time):
        DummyJob.__init__(self, aggregator, sleep_time)
        
    def compute(self):
        # import all classes
        from modshogun import DirectSparseLinearSolver
        from modshogun import RealSparseMatrixOperator
        from modshogun import Statistics
        from modshogun import CGMShiftedFamilySolver
        from modshogun import LanczosEigenSolver
        from modshogun import LogDetEstimator
        from modshogun import LogRationalApproximationCGM
        from modshogun import ProbingSampler
        from modshogun import RealSparseMatrixOperator
        from modshogun import SerialComputationEngine
        
        # create empty instances once for all classes
        print DirectSparseLinearSolver().get_name()
        print RealSparseMatrixOperator().get_name()
        print CGMShiftedFamilySolver().get_name()
        print LanczosEigenSolver().get_name()
        print LogDetEstimator().get_name()
        print LogRationalApproximationCGM().get_name()
        print ProbingSampler().get_name()
        print RealSparseMatrixOperator().get_name()
        print SerialComputationEngine().get_name()
        
        DummyJob.compute(self)