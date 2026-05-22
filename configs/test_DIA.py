import tensorflow as tf

from uNAS.config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from dataset.DIA_dataset import DIA_Dataset
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch, BayesOpt

def get_DIA_setup(classes=[0,1],
                  error_bound=0.2, peak_mem_bound=5_000_000,
                  model_size_bound=10_000_000, mac_bound=5_000_000):
    return {
        'config': get_DIA_config(classes=classes,
                                  error_bound=error_bound, peak_mem_bound=peak_mem_bound,
                                  model_size_bound=model_size_bound, mac_bound=mac_bound),
        'name': 'DIA_Test',
        'load_from': None,
        'save_every': 10,
        'seed': 0
        }


def get_DIA_config(classes=[0,1],
                   error_bound=0.2, peak_mem_bound=5_000_000,
                   model_size_bound=10_000_000, mac_bound=5_000_000):
    training_config = TrainingConfig(
        dataset = DIA_Dataset(classes = classes), 
        optimizer = lambda: tf.optimizers.Adam(learning_rate=0.0001),
        callbacks = lambda: [
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1, min_lr=0.000001),  
                            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0.5, verbose=1, restore_best_weights=True), 
                            # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=1, restore_best_weights=True), 
                            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.005, verbose=1, restore_best_weights=True), 
                            tf.keras.callbacks.TerminateOnNaN()
                            ],
        epochs = 30, 
        batch_size = 16
    )

    bound_config = BoundConfig(
        error_bound=error_bound,
        peak_mem_bound=peak_mem_bound,
        model_size_bound=model_size_bound,
        mac_bound=mac_bound,
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
        search_space = Cnn1DSearchSpace(),
        checkpoint_dir = "artifacts/DIA_512smpl_11072024",
        rounds = 1
    )


    model_saver_config = ModelSaverConfig(
        save_criteria = "all"
        #save_path = "artifacts/DIA_512smpl_11072024/model_saver"
    )
    
    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config,
        'serialized_dataset': False
        }
