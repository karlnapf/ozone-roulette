ozone-roulette
==============

Distributed implementation of Russian Roulette for a GMRF on ozone data. Has been observed to scale up to >1000 computing nodes. Please note that this is a proof-of-concept implementation.

###Dependencies:
The following libraries and files need to be available.

####Data:
Ozone data has to be in $HOME/data/ozone/ - ask author if you need it

####Python libraries: (Need to be in PYTHONPATH):
 * https://github.com/karlnapf/kameleon-mcmc
 * https://github.com/karlnapf/independent-jobs

There are two different backends for batch-type cluster computing systems: PBS and SGE. See ozone/scripts/ folder for scripts that can be used for experiments.

####For estimating log-determinants:
Shogun Machine Learning Toolbox - http://shogun-toolbox.org/
Needs to be compiled with:
logdet framework, which depends on eigen3, lapack, and colpack
python_modular language bindings, which depend on python and numpy

For sparse Cholesky (optional, can be done with Shogun):
Python's sparse cholmod package - http://pythonhosted.org/scikits.sparse/cholmod.html



Written (W) 2013-2014 Heiko Strathmann. See License for copyright.

