import tensorflow as tf
from uNAS.config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from dataset import SpeechCommandsDataset
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch
from functools import partial


def get_speechcommands_setup(input_size=(49, 13), batch_size=512, serialized=False, fix_seeds=False):
       
    return {
        'config': get_speechcommands_config(input_size=input_size, batch_size=batch_size, serialized=serialized, fix_seeds=fix_seeds),
        'name': 'speechcommands_cnn1d_search',
        'load_from': None,
        'save_every': 5,
        'seed': 0
    }

def get_speechcommands_config(input_size=(49, 13), batch_size=512, serialized=False, fix_seeds=False):
    training_config = TrainingConfig(
        dataset=partial(SpeechCommandsDataset, 
                       data_dir='./speech_dataset', 
                       sample_rate=16000,
                       num_mfcc=input_size[1],
                       fix_seeds=fix_seeds),
        optimizer="adam",
        batch_size=batch_size,
        epochs=50,
        callbacks=lambda: [
            tf.keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN()
        ],
        serialized_dataset=serialized
    )

    search_algorithm = AgingEvoSearch

    search_config = AgingEvoConfig(
        search_space=Cnn1DSearchSpace(),
        serialized_dataset=serialized,
        checkpoint_dir="artifacts/1dcnn_speechcommands",
        max_parallel_evaluations=1,
        rounds=2000
    )

    bound_config = BoundConfig(
        error_bound=0.2,
        peak_mem_bound=250_000,
        model_size_bound=500_000,
        mac_bound=1_000_000
    )

    model_saver_config = ModelSaverConfig(
        save_criteria="all"
    )

    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config,
        'serialized_dataset': serialized
    }
