import tensorflow as tf

from config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from dataset.Z24_dataset import Z24_Dataset
from cnn1d import Cnn1DSearchSpace
from search_algorithms import AgingEvoSearch, BayesOpt


training_config = TrainingConfig(
    dataset = Z24_Dataset(classes = ['01', '03'], windows_length=65536, path= 'C:\\Dottorato\\Z24\\DatasetPDT', fix_seeds = False), # , '04', '05', '06'
    optimizer = lambda: tf.optimizers.Adam(learning_rate=0.001),
    callbacks = lambda: [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)],
    epochs = 1, # 75 
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
    search_space= Cnn1DSearchSpace(),  # CnnSearchSpace(), # MlpSearchSpace(),  #  
    starting_points=10,
    checkpoint_dir="artifacts/cnn_test_dummy_dataset_model_saver"
)
'''


search_algorithm = AgingEvoSearch

search_config = AgingEvoConfig(
    search_space = Cnn1DSearchSpace(),
    checkpoint_dir = "artifacts/cnn_test_dummy_dataset_model_saver",
    rounds = 1 # 500
)



model_saver_config = ModelSaverConfig(
    save_criteria = "pareto",
    save_path = "artifacts/cnn_test_dummy_dataset/model_saver"
)
