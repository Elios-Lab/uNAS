import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from uNAS import uNAS
 
#from uNAS.examples import get_example_2d_unas_setup
#from configs.test_Z24 import get_Z24_setup
#from configs.test_dummy_dataset import get_dummy_2D_setup
#from configs.test_DIA import get_DIA_setup
from configs.cnn_wakeviz import get_wakeviz_setup
from configs.test_SR import get_speechcommands_setup
from configs.test_HAR import get_HAR_setup
from configs.test_regression import get_REG_setup


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Driver")
    
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Input shape should be (timesteps, features, channels)
    unas_setup = get_REG_setup()
    unas = uNAS(unas_setup, logger)
    unas.run()


if __name__ == "__main__":
    main()
