import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from pathlib import Path

from uNAS import uNAS

from uNAS.examples import get_example_1d_unas_setup, get_example_2d_unas_setup, get_example_mlp_unas_setup



def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Driver")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    unas_config = get_example_mlp_unas_setup()

    unas = uNAS(unas_config, logger)

    unas.run()


if __name__ == "__main__":
    main()
