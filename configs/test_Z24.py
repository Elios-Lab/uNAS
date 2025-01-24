import tensorflow as tf

from uNAS.config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from dataset.Z24_dataset import Z24_Dataset
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch, BayesOpt

def get_Z24_setup(classes = ['01', '03', '04', '05', '06'], windows_length=512):
    return {
        'config': get_Z24_config(classes=classes,windows_length=windows_length),
        'name': 'Z24_Test',
        'load_from': None,
        'save_every': 5,
        'seed': 0
        }


def get_Z24_config(classes = ['01', '03', '04', '05', '06'], windows_length=512):
    training_config = TrainingConfig(
        dataset = Z24_Dataset(classes = classes, windows_length=windows_length, path= rf'/mnt/c/Users/Matteo/Desktop/Dottorato/BridgeZ24_1/DatasetPDT', fix_seeds = True), #  windows_length=8192,
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
        checkpoint_dir = "artifacts/Z24_512smpl_11072024",
        rounds = 1
    )


    model_saver_config = ModelSaverConfig(
        save_criteria = "all"
        #save_path = "artifacts/Z24_512smpl_11072024/model_saver"
    )
    
    return {
        'training_config': training_config,
        'bound_config': bound_config,
        'search_algorithm': search_algorithm,
        'search_config': search_config,
        'model_saver_config': model_saver_config,
        'serialized_dataset': False
        }
