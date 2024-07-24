import os
import argparse
import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from pathlib import Path


from uNAS.search_algorithms import BayesOpt

from uNAS import uNAS



def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("Driver")

    parser = argparse.ArgumentParser("uNAS Search")
    parser.add_argument("config_file", type=str, help="A config file describing the search parameters")
    parser.add_argument("--name", type=str, help="Experiment name (for disambiguation during state saving)")
    parser.add_argument("--load-from", type=str, default=None, help="A search state file to resume from")
    parser.add_argument("--save-every", type=int, default=5, help="After how many search steps to save the state")
    parser.add_argument("--seed", type=int, default=0, help="A seed for the global NumPy and TensorFlow random state")
    args = parser.parse_args()

    unas = uNAS(args, log)

    unas.run()


if __name__ == "__main__":
    main()
