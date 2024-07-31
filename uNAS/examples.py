"""
This module contains functions to get examples of configurations for the uNAS framework.

In particular, it contains examples for the 1D Convolutional Neural Networks, 2D CNN, and Multi-Layer Perceptrons modules.
The examples also incorporate pruning configurations.

More details about the elements of the configuration can be found in the uNAS.config module.

The module contains the following functions:
- get_template_setup: returns a template setup for the uNAS
- get_example_1d_unas_setup: returns an example starting setup for uNAS with 1D CNN
- get_example_2d_unas_setup: returns an example starting setup for uNAS with 2D CNN
- get_example_mlp_unas_setup: returns an example starting setup for uNAS with MLP

- get_template_config: returns a template configuration for the uNAS framework
- get_example_1dcnn_aging_config: returns an example configuration for the 1D CNN module
- get_example_2dcnn_aging_config: returns an example configuration for the 2D CNN module
- get_example_mlp_aging_config: returns an example configuration for the MLP module
- get_example_1dcnn_aging_pruning_config: returns an example configuration for the 1D CNN module with pruning
"""

import tensorflow as tf
from uNAS.dummy_datasets import Dummy1D, Dummy2D, DummyTabular
from uNAS.search_algorithms import AgingEvoSearch
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.cnn2d import Cnn2DSearchSpace
from uNAS.mlp import MlpSearchSpace


from uNAS.config import PruningConfig, TrainingConfig, BayesOptConfig, AgingEvoConfig, BoundConfig, ModelSaverConfig

def get_template_setup():
    """
    This function returns a template setup for the uNAS.

    This setup is filled with None values and should be used as a template to create new configurations.

    The nas module setup is a dictionary with the following keys
    and values:
    - config: dict, the configuration for the NAS module
    - name: str, name of the experiment
    - load_from: str, Optional. Path to the search state file to resume from
    - save_every: int, after how many search steps to save the state
    - seed: int, a seed for the global NumPy and TensorFlow random state

    The config is a dictionary with the following keys and values:
    - training_config: TrainingConfig, the configuration for the training process
    - bound_config: BoundConfig, the configuration for the search bounds
    - search_algorithm: SearchAlgorithm, the search algorithm to use
    - search_config: SearchConfig, the configuration for the search algorithm
    - model_saver_config: ModelSaverConfig, the configuration for the model saver


    """

    return {
        'config': get_template_config(),
        'name': None,
        'load_from': None,
        'save_every': None,
        'seed': None
        }

def get_example_1d_unas_setup():
    """
    Returns an example starting setup for uNAS.

    The nas module setup is a dictionary with the following keys
    and values:
    - config: dict, the configuration for the NAS module
    - name: str, name of the experiment
    - load_from: str, Optional. Path to the search state file to resume from
    - save_every: int, after how many search steps to save the state
    - seed: int, a seed for the global NumPy and TensorFlow random state

    This setup configure the NAS to run on a 1D CNN module with the Dummy1D dataset.

    The setup uses the AgingEvoSearch algorithm for the search., it saves the search state every 5 steps and uses a seed of 0 for the random state.

    """
    return {
        'config': get_example_1dcnn_aging_config(),
        'name': 'test_uNAS_module',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }

def get_example_2d_unas_setup():
    """
    Returns an example starting setup for uNAS.

    The nas module setup is a dictionary with the following keys
    and values:
    - config: dict, the configuration for the NAS module
    - name: str, name of the experiment
    - load_from: str, path to the search state file to resume from
    - save_every: int, after how many search steps to save the state
    - seed: int, a seed for the global NumPy and TensorFlow random state

    This setup configure the NAS to run on a 2D CNN module with the Dummy2D dataset.

    The setup uses the AgingEvoSearch algorithm for the search., it saves the search state every 5 steps and uses a seed of 0 for the random state.

    """
    return {
        'config': get_example_2dcnn_aging_config(),
        'name': 'test_uNAS_module',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }

def get_example_mlp_unas_setup():
    """
    Returns an example starting setup for uNAS.

    The nas module setup is a dictionary with the following keys
    and values:
    - config: dict, the configuration for the NAS module
    - name: str, name of the experiment
    - load_from: str, path to the search state file to resume from
    - save_every: int, after how many search steps to save the state
    - seed: int, a seed for the global NumPy and TensorFlow random state

    This setup configure the NAS to run on a MLP module with the DummyTabular dataset.

    The setup uses the AgingEvoSearch algorithm for the search., it saves the search state every 5 steps and uses a seed of 0 for the random state.

    """
    return {
        'config': get_example_mlp_aging_config(),
        'name': 'test_uNAS_module',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }



def get_template_config():
    """
    This function returns a template configuration for the uNAS framework.

    This configuration is filled with None values and should be used as a template to create new configurations.

    The config is a dictionary with the following keys and values:
    - training_config: TrainingConfig, the configuration for the training process
    - bound_config: BoundConfig, the configuration for the search bounds
    - search_algorithm: SearchAlgorithm, the search algorithm to use
    - search_config: SearchConfig, the configuration for the search algorithm
    - model_saver_config: ModelSaverConfig, the configuration for the model saver
    """
    training_config = TrainingConfig(dataset=None, optimizer=None, callbacks=None, epochs=None, batch_size=None)
    bound_config = BoundConfig()
    search_algorithm = None
    search_config = None
    model_saver_config = ModelSaverConfig()

    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config
        }

def get_example_1dcnn_aging_config():
    """
    Returns an example configuration for the 1D CNN module.

    This configuration allow the user to run NAS on 1D CNNs for time series classification.

    It uses the Dummy2D dataset for testing purposes.
    
    The search algorithm is set to AgingEvoSearch.
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
        checkpoint_dir = "artifacts/1dcnn_dummy1d_dataset",
        rounds = 150
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

def get_example_2dcnn_aging_config():
    """
    Returns an example configuration for the 2D CNN module.

    This configuration allow the user to run NAS on 2D CNNs for image classification.

    It uses the Dummy2D dataset for testing purposes.

    The search algorithm is set to AgingEvoSearch.
    """
    training_config = TrainingConfig( 
        dataset = Dummy2D(img_shape=(32, 32, 3), num_classes=10, length=100),
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
        search_space = Cnn2DSearchSpace(),
        checkpoint_dir = "artifacts/2dcnn_dummy1d_dataset",
        rounds = 150
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

def get_example_mlp_aging_config():
    """
    Returns an example configuration for the MLP module.

    This configuration allow the user to run NAS on MLPs for tabular data classification.

    It uses the Dummy1D dataset for testing purposes.

    The search algorithm is set to AgingEvoSearch.
    """
    training_config = TrainingConfig( 
        dataset = DummyTabular(num_features=5, num_classes=5, length=1000),
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
        search_space = MlpSearchSpace(),
        checkpoint_dir = "artifacts/mlp_dummy1d_dataset",
        rounds = 150
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

def get_example_1dcnn_aging_pruning_config():
    """
    Returns an example configuration for the 1D CNN module with pruning.

    This configuration allow the user to run NAS on 1D CNNs for time series classification.

    It uses the Dummy1D dataset for testing purposes.
    
    The search algorithm is set to AgingEvoSearch.
    """
    pruning_config = PruningConfig(
        structured= False,
        start_pruning_at_epoch= 20,
        finish_pruning_by_epoch= 45,
        min_sparsity= 0.2,
        max_sparsity= 0.4,
        )
    
    training_config = TrainingConfig( 
        dataset = Dummy1D(samples_per_second= 1000, duration=1, length=100, difficulty=1, num_classes = 6),
        optimizer = lambda: tf.optimizers.Adam(learning_rate=0.001),
        callbacks = lambda: [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)],
        epochs = 75,
        batch_size= 16,
        pruning = pruning_config
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
        checkpoint_dir = "artifacts/1dcnn_dummy1d_dataset",
        rounds = 150
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



'''

def lr_schedule(epoch):
    if 0 <= epoch < 35:
        return 0.01
    return 0.005


training_config = TrainingConfig(
    dataset=Chars74K("/datasets/chars74k", img_size=(48, 48)),
    epochs=60,
    batch_size=80,
    optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.01, momentum=0.9, weight_decay=0.0001),
    callbacks=lambda: [LearningRateScheduler(lr_schedule)],
)

search_config = BayesOptConfig(
    search_space=Cnn2DSearchSpace(dropout=0.15),
    checkpoint_dir="artifacts/cnn_chars74k"
)

'''