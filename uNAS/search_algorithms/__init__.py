"""
This module contains the search algorithms for the uNAS framework.
The search algorithms are used to find the best architecture for the dataset.

The module contains the following classes:
- AgingEvoSearch: Aging Evolution Search algorithm
- BayesOpt: Bayesian Optimisation Search algorithm

"""

from .aging_evolution import AgingEvoSearch
from .bayesian_optimisation import BayesOpt
