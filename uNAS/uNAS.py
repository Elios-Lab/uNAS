# This class constitutes the entry point for the μNAS tool. It is responsible for parsing the command line arguments and
# calling the appropriate functions to perform the requested operations.


import os
import argparse
import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from pathlib import Path

from uNAS.search_algorithms import BayesOpt

from uNAS.model_saver import ModelSaver

import time

from uNAS.config import DistillationConfig, PruningConfig, TrainingConfig, BayesOptConfig, AgingEvoConfig, BoundConfig, ModelSaverConfig
from uNAS.search_algorithms import AgingEvoSearch, BayesOpt
from uNAS.dataset import Dataset

class uNAS:
    """
    uNAS Module
    ===========
    The main class for the μNAS tool. This class is responsible for parsing the arguments and calling the appropriate functions to perform the requested operations.

    It requires a configuration dictionary and a logger object to log the operations.

    The configuration dictionary must contain the following keys and values:
    - config_file: str, path to the configuration file
    - name: str, name of the experiment
    - load_from: str, path to the search state file to resume from
    - save_every: int, after how many search steps to save the state
    - seed: int, a seed for the global NumPy and TensorFlow random state

    It exposes the following methods:
    - run: Run the μNAS tool
    - validate_setup: Validate the configuration parameters

    Usage example:
    --------------

    from uNAS import uNAS

    from uNAS.types import get_example_unas_config

    import logging

    config = get_example_unas_config()

    config['config_file'] = '{your configuration file path}'

    logger = logging.getLogger('Driver')

    unas = uNAS(config, logger)

    unas.run()
    """

    _configs = None
    _model_saver = None
    _search_space = None
    _dataset = None
    _search = None
    _log = None
    _unas_config = None


    def __init__(self, unas_config, log = None):
        self._log = log
        self._unas_config = unas_config
        self._configs = unas_config["config"]

        self._validate_setup(self._unas_config)
        self._configure_seeds(self._unas_config["seed"])

    
    def _configure_seeds(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)


    def _configure_search_algorithm(self):
        if "search_algorithm" not in self._configs:
            algo = BayesOpt
        else:
            algo = self._configs["search_algorithm"]
    
        search_space = self._configs["search_config"].search_space
        dataset = self._configs["training_config"].dataset
        search_space.input_shape = dataset.input_shape
        search_space.num_classes = dataset.num_classes

        ckpt_path = self._configs["search_config"].checkpoint_dir

        model_saver = ModelSaver(self._configs["model_saver_config"], self._configs["bound_config"], ckpt_path=ckpt_path)

        self._search = algo(experiment_name=self._unas_config["name"] or "search",
                    search_config=self._configs["search_config"],
                    training_config=self._configs["training_config"],
                    
                    bound_config=self._configs["bound_config"], model_saver=model_saver)
        
        self._model_saver = model_saver

    def run(self):
        self._configure_search_algorithm()

        if self._unas_config["save_every"] <= 0:
            raise argparse.ArgumentTypeError("Value for '--save-every' must be a positive integer.")

        if self._unas_config.get("load_from", False) and not os.path.exists(self._unas_config["load_from"]):
            self._log.warning("Search state file to load from is not found, the search will start from scratch.")
            self._unas_config["load_from"] = None

        self._search.search(load_from=self._unas_config["load_from"], save_every=self._unas_config["save_every"])

        self._log.info("Wait for the last model to be evaluated")

        # Wait for the last model to be evaluated and saved
        time.sleep(30)      

        self._log.info("Search complete")
        self._log.info("Dumping models")
        self._model_saver.save_models()



    #@staticmethod
    def _validate_setup(self, setup):
        """
        Validate the setup parameters.
        """
        self._log.info("Validating setup parameters")


        if setup["name"] is None:
            raise Exception("Name must be provided.")
        if type(setup["name"]) is not str:
            raise Exception("Name must be a string.")
        
        if setup["load_from"] is not None and not os.path.exists(setup["load_from"]):
            raise Exception("Search state file to load from is not found.")
        
        if type(setup["save_every"]) is not int:
            raise Exception("Value for 'save-every' must be an integer.")
        if setup["save_every"] <= 0:
            raise Exception("Value for 'save-every' must be a positive integer.")
        
        if type(setup["seed"]) is not int:
            raise Exception("Value for 'seed' must be an integer.")
        if setup["seed"] < 0:
            raise Exception("Value for 'seed' must be a non-negative integer.")
        
        if "config" not in setup:
            raise Exception("Configuration must be provided.")
        if type(setup["config"]) is not dict:
            raise Exception("Configuration must be a dictionary.")
        if "search_config" not in setup["config"]:
            raise Exception("Search configuration must be provided.")
        if "training_config" not in setup["config"]:
            raise Exception("Training configuration must be provided.")
        if "model_saver_config" not in setup["config"]:
            raise Exception("Model saver configuration must be provided.")
        if "bound_config" not in setup["config"]:
            raise Exception("Bound configuration must be provided.")
        if "search_algorithm" not in setup["config"]:
            raise Exception("Search algorithm must be provided.")
        
        if type(setup["config"]["search_config"]) is not AgingEvoConfig and type(setup["config"]["search_config"]) is not BayesOptConfig:
            raise Exception("Search configuration must be an instance of class AgingEvoConfig or BayesOptConfig.")
        
        if type(setup["config"]["training_config"]) is not TrainingConfig:
            raise Exception("Training configuration must be an instance of class TrainingConfig.")
        
        if type(setup["config"]["model_saver_config"]) is not ModelSaverConfig:
            raise Exception("Model saver configuration must be an instance of class ModelSaverConfig.")
        
        if type(setup["config"]["bound_config"]) is not BoundConfig:
            raise Exception("Bound configuration must be an instance of class BoundConfig.")
        
        if setup["config"]["search_algorithm"] is not AgingEvoSearch and setup["config"]["search_algorithm"] is not BayesOpt:
            raise Exception("Search algorithm must be an instance of class AgingEvoSearch or BayesOpt.")
        
        if setup["config"]["training_config"].dataset is None:
            raise Exception("Dataset must be provided in training configuration.")
        
        if not isinstance(setup["config"]["training_config"].dataset, Dataset):
            raise Exception("Dataset must be an instance of class Dataset.")
        
        if setup["config"]["training_config"].distillation is not None:
            if not isinstance(setup["config"]["training_config"].distillation, DistillationConfig):
                raise Exception("Distillation  must be an instance of class DistillationConfig or None.")
            
        if setup["config"]["training_config"].pruning is not None:
            if not isinstance(setup["config"]["training_config"].pruning_config, PruningConfig):
                raise Exception("Pruning must be an instance of class PruningConfig or None.")
            

        self._log.info("Setup parameters are valid!")
        

        
        

    

