"""
uNAS Module
===========

This module provides tools and functionalities for running Neural Architecture Search (NAS) using various algorithms.

Getting Started:
----------------
To get started with uNAS, we recommend looking at the example configurations provided. These can be found in the `configs` directory or accessed programmatically using the `get_example_unas_config` function.

Example:
--------
Here's a quick example to get a default configuration for uNAS:

    from uNAS import get_example_unas_config
    unas_config = get_example_unas_config()
    print(unas_config)

For more detailed documentation and advanced usage, please refer to our online documentation at [Link to Documentation].

"""

from .config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from .dataset import Dataset
from .uNAS import uNAS
from .search_algorithms import AgingEvoSearch, BayesOpt
from .cnn import CnnSearchSpace
from .mlp import MlpSearchSpace
from .cnn1d import Cnn1DSearchSpace
from .search_space import SearchSpace
from .model_saver import ModelSaver
from .utils import generate_nth_id, NumpyEncoder, num_gpus, debug_mode, Scheduler, quantised_accuracy, copy_weight
from .types import get_example_unas_config