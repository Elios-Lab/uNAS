import tensorflow as tf

from uNAS.config import TrainingConfig, BayesOptConfig, BoundConfig, AgingEvoConfig, ModelSaverConfig
from uNAS.dummy_datasets import Dummy1D
from uNAS.dummy_datasets import Dummy2D
from uNAS.mlp import MlpSearchSpace
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.cnn import CnnSearchSpace
from uNAS.search_algorithms import AgingEvoSearch, BayesOpt


training_config = TrainingConfig(
    dataset = Dummy1D(samples_per_second= 1000, duration=1, length=100, difficulty=1, num_classes = 6), # Dummy2D((10,10,1),2,50),  #
    optimizer = lambda: tf.optimizers.Adam(learning_rate=0.001),
    callbacks = lambda: [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)],
    epochs = 1, # 75 
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
    search_space= Cnn1DSearchSpace(),  # CnnSearchSpace(), # MlpSearchSpace(),  #  
    starting_points=10,
    checkpoint_dir="artifacts/cnn_test_dummy_dataset_model_saver"
)
'''


search_algorithm = AgingEvoSearch

search_config = AgingEvoConfig(
    search_space = Cnn1DSearchSpace(),
    checkpoint_dir = "artifacts/cnn_test_dummy_dataset_model_saver",
    rounds = 500
)


model_saver_config = ModelSaverConfig(
    save_criteria = "all",
)
