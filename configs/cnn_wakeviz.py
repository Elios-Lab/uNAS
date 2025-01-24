import tensorflow as tf
from uNAS.config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from dataset import WV_Dataset
from uNAS.cnn2d import Cnn2DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch, BayesOpt
from functools import partial


def get_wakeviz_setup(input_size = (50,50) , batch_size = 512, serialized=False, fix_seeds = False):
    
    return {
        'config': get_wakeviz_config(input_size=input_size, batch_size=batch_size, serialized=serialized, fix_seeds=fix_seeds),
        'name': 'wakeviz_cnn_test',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }

def get_wakeviz_config(input_size = (50,50) , batch_size = 512, serialized=False, fix_seeds = False):

    training_config = TrainingConfig(
        dataset=partial(WV_Dataset, 
                    data_dir='/media/pigo/22F2EE2BF2EE0341/wake_vision/', 
                    input_shape=input_size, 
                    fix_seeds=fix_seeds),
        optimizer=lambda: tf.optimizers.SGD(learning_rate=0.001, weight_decay=5e-5),
        batch_size=batch_size,
        epochs=10,
        callbacks=lambda: [tf.keras.callbacks.EarlyStopping(patience=15, verbose=1),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1,  min_lr=0.000001),
                        tf.keras.callbacks.TerminateOnNaN()],
        serialized_dataset=serialized
    )

    search_algorithm = AgingEvoSearch

    search_config = AgingEvoConfig(
        search_space=Cnn2DSearchSpace(),
        serialized_dataset=serialized,
        checkpoint_dir="artifacts/cnn_vwviz",
        max_parallel_evaluations=1,
        rounds = 2000
    )

    bound_config = BoundConfig(
    error_bound = 0.3,
    peak_mem_bound = 250_000, 
    model_size_bound = 450_000,
    mac_bound = 3_500_000
    )

    model_saver_config = ModelSaverConfig(
        save_criteria = "all"
    )

    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config,
        'serialized_dataset': serialized
        }