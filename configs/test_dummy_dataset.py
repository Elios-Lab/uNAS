import tensorflow as tf

from uNAS.config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from uNAS.dummy_datasets import Dummy1D
from uNAS.dummy_datasets import Dummy2D
from uNAS.mlp import MlpSearchSpace
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.cnn2d import Cnn2DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch, BayesOpt


def get_dummy_2D_setup():
    return {
        'config': get_dummy_2D_config(),
        'name': 'dummy_2D_test',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }


def get_dummy_2D_config():
    training_config = TrainingConfig(
        dataset = Dummy2D((10,10,1),2,50), 
        optimizer = lambda: tf.optimizers.SGD(learning_rate=0.001, weight_decay=5e-5),
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



    '''
    search_algorithm = BayesOpt

    search_config = BayesOptConfig(
        search_space= Cnn1DSearchSpace(),  # Cnn2DSearchSpace(), # MlpSearchSpace(),  #  
        starting_points=10,
        checkpoint_dir="artifacts/cnn_test_dummy_dataset_model_saver"
    )
    '''


    search_algorithm = AgingEvoSearch

    search_config = AgingEvoConfig(
        search_space = Cnn2DSearchSpace(),
        checkpoint_dir = "artifacts/cnn_test_dummy_dataset_model_saver",
        rounds = 500
    )


    model_saver_config = ModelSaverConfig(
        save_criteria = "all",
    )

    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config,
        'serialized_dataset': False
        }
