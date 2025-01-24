import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from pathlib import Path

from uNAS import uNAS
 
from uNAS.examples import get_example_2d_unas_setup
from configs.cnn_wakeviz import get_wakeviz_setup
from configs.test_dummy_dataset import get_dummy_2D_setup

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Driver")
    
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    unas_setup = get_wakeviz_setup(input_size = (50,50) , batch_size = 8, fix_seeds = False, serialized=True)

    unas = uNAS(unas_setup, logger)

    unas.run()


if __name__ == "__main__":
    main()
