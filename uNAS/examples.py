"""
This module contains functions to get examples of configurations for the uNAS framework.

The module contains the following functions:
- get_example_cnn_config: returns an example configuration for the CNN module
- get_example_cnn1d_config: returns an example configuration for the CNN1D module
- get_example_mlp_config: returns an example configuration for the MLP module
- get_example_unas_config: returns an example starting configuration for uNAS
"""

import tensorflow as tf
from uNAS.dummy_datasets import Dummy1D, Dummy2D
from uNAS.search_algorithms import AgingEvoSearch
from uNAS.cnn1d import Cnn1DSearchSpace


from uNAS.config import DistillationConfig, PruningConfig, TrainingConfig, BayesOptConfig, AgingEvoConfig, BoundConfig, ModelSaverConfig

def get_example_unas_config():
    """
    Returns an example starting configuration for uNAS.

    The nas module configuration is a dictionary with the following keys
    and values:
    - config_file: str, path to the configuration file
    - name: str, name of the experiment
    - load_from: str, path to the search state file to resume from
    - save_every: int, after how many search steps to save the state
    - seed: int, a seed for the global NumPy and TensorFlow random state
    """
    return {
        'config': get_example_1dcnn_config(),
        'name': 'test_uNAS_module',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }


def get_example_1dcnn_config():
    """
    Returns an example configuration for the 1D CNN module.

    This configuration allow the user to run NAS on 1D CNNs for time series classification.

    It uses the Dummy2D dataset for testing purposes.
    """
    training_config = TrainingConfig( 
        dataset = Dummy1D(samples_per_second= 1000, duration=1, length=100, difficulty=1, num_classes = 6),
        optimizer = lambda: tf.optimizers.Adam(learning_rate=0.001),
        callbacks = lambda: [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)],
        epochs = 75,
        batch_size= 16
        )
    bound_config = BoundConfig(
        error_bound = 0.4,
        peak_mem_bound = 20000,
        model_size_bound = 50000,
        mac_bound = 30000
        )
    search_algorithm = AgingEvoSearch

    search_config = AgingEvoConfig(
        search_space = Cnn1DSearchSpace(),
        checkpoint_dir = "artifacts/cnn_test_dummy_dataset_model_saver",
        rounds = 500
        )
    
    model_saver_config = ModelSaverConfig(        
    save_criteria = "none",
    )
    
    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config
        }



    