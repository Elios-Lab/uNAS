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

    _configs = None
    _model_saver = None
    _search_space = None
    _dataset = None
    _search = None
    _log = None


    def __init__(self, configs, log):
        self._configs = configs
        self._log = log


    
    def _configure_seeds(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _configure_gpus(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def _configure_search_algorithm(self):
        if "search_algorithm" not in self._configs:
            algo = BayesOpt
        else:
            algo = self._configs["search_algorithm"]
    
        search_space = self._configs["search_config"].search_space
        dataset = self._configs["training_config"].dataset
        search_space.input_shape = dataset.input_shape
        search_space.num_classes = dataset.num_classes

        model_saver = ModelSaver(self._configs["model_saver_config"], self._configs["bound_config"])

        self._search = algo(experiment_name="args.name" or "search",
                    search_config=self._configs["search_config"],
                    training_config=self._configs["training_config"],
                    
                    bound_config=self._configs["bound_config"], model_saver=model_saver)
        
        self._model_saver = model_saver

    def run(self, args):
        #self._configure_gpus()
        self._configure_seeds(args.seed)
        self._configure_search_algorithm()

        if args.save_every <= 0:
            raise argparse.ArgumentTypeError("Value for '--save-every' must be a positive integer.")

        if args.load_from and not os.path.exists(args.load_from):
            self._log.warning("Search state file to load from is not found, the search will start from scratch.")
            args.load_from = None

        self._search.search(load_from=args.load_from, save_every=args.save_every)

        print("Search complete")

        # Wait for the last model to be evaluated and saved
        time.sleep(30)
        print("Dumping models")
        self._model_saver.save_models()

    

