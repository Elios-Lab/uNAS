import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

from pathlib import Path

from uNAS import uNAS

from uNAS.types import get_example_unas_config



def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Driver")



    unas_config = get_example_unas_config()
    unas_config['config_file'] = 'configs/test_dummy_dataset.py'
    unas_config['name'] = 'test_uNAS_module'
    unas_config['load_from'] = None
    unas_config['save_every'] = 5
    unas_config['seed'] = 0

    uNAS.validate_config(unas_config)

    unas = uNAS(unas_config, logger)

    unas.run()


if __name__ == "__main__":
    main()
