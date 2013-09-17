ozone-roulette
==============

Implementation of Russian Roulette for a GMRF on ozone data.

Dependencies 

Data:
Ozone data has to be in $HOME/data/ozone/ - ask author if you need it

Python libraries: (Need to be in PYTHONPATH):
https://github.com/karlnapf/kameleon-mcmc
https://github.com/karlnapf/independent-jobs

For estimating log-determinants:
Shogun Machine Learning Toolbox - http://shogun-toolbox.org/
Needs to be compiled with:
logdet framework, which depends on eigen3, lapack, and colpack
python_modular language bindings, which depend on python and numpy

For sparse Cholesky (can be done with Shogun):
Python's sparse cholmod package - http://pythonhosted.org/scikits.sparse/cholmod.html

Written (W) 2013 Heiko Strathmann under GPL3
