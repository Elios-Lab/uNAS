import os
import argparse
import logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from pathlib import Path


from uNAS.search_algorithms import BayesOpt

from uNAS import uNAS

import time


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


    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if args.save_every <= 0:
        raise argparse.ArgumentTypeError("Value for '--save-every' must be a positive integer.")

    configs = {}
    exec(Path(args.config_file).read_text(), configs)

    uNAS(configs, log).run(args)


if __name__ == "__main__":
    main()
