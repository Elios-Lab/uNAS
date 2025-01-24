import tensorflow as tf

from uNAS.config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from dataset.SR_dataset import SR_Dataset
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch, BayesOpt

def get_SR_setup(classes = [0,1,2,3,4,5,6,7,8,9,10,11]):
    return {
        'config': get_SR_config(classes=classes),
        'name': 'SR_Test',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }


def get_SR_config(classes = [0,1,2,3,4,5,6,7,8,9,10,11]):
    training_config = TrainingConfig(
        dataset = SR_Dataset(classes = classes), 
        optimizer = lambda: tf.optimizers.Adam(learning_rate=0.0005),
        callbacks = lambda: [
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1, min_lr=0.000001),  
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, min_delta=0.5, verbose=1, restore_best_weights=True), 
                            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=150, verbose=1, restore_best_weights=True), 
                            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=75, min_delta=0.005, verbose=1, restore_best_weights=True), 
                            tf.keras.callbacks.TerminateOnNaN()
                            ],
        epochs = 1000, 
        batch_size = 8
    )

    bound_config = BoundConfig(
    error_bound = 0.3,
    peak_mem_bound = 5000000, 
    model_size_bound = 10000000,
    mac_bound = 5000000
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
        checkpoint_dir = "artifacts/SR_512smpl_11072024",
        rounds = 1
    )


    model_saver_config = ModelSaverConfig(
        save_criteria = "all"
        #save_path = "artifacts/SR_512smpl_11072024/model_saver"
    )
    
    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config,
        'serialized_dataset': False
        }
