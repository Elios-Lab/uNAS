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
        self.validate_setup(unas_config)
        self._configure_seeds(unas_config["seed"])
        self._log = log
        self._unas_config = unas_config
        self._configs = unas_config["config"]

    
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


    #this is a static method to validate the configuration params
    @staticmethod
    def validate_setup(config):
        """
        Validate the configuration parameters.
        """

        if config["name"] is None:
            raise Exception("Name must be provided.")
        if type(config["name"]) is not str:
            raise Exception("Name must be a string.")
        if config["load_from"] is not None and not os.path.exists(config["load_from"]):
            raise Exception("Search state file to load from is not found.")
        if type(config["save_every"]) is not int:
            raise Exception("Value for 'save-every' must be an integer.")
        if config["save_every"] <= 0:
            raise Exception("Value for 'save-every' must be a positive integer.")
        if type(config["seed"]) is not int:
            raise Exception("Value for 'seed' must be an integer.")
        if config["seed"] < 0:
            raise Exception("Value for 'seed' must be a non-negative integer.")
        

    

