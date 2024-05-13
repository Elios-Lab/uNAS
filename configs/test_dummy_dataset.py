import tensorflow as tf

from config import TrainingConfig, BayesOptConfig, BoundConfig
from dataset.dummy_waveform import DummyWaveform
from dataset.dummy import Dummy
from mlp import MlpSearchSpace
from cnn1d import Cnn1DSearchSpace
from cnn import CnnSearchSpace


training_config = TrainingConfig(
    dataset=  DummyWaveform(samples_per_second= 1000, duration=1, length=5000), # Dummy((10,10,1),2,50),  #
    optimizer=lambda: tf.optimizers.Adam(learning_rate=0.001),
    callbacks=lambda: [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)]
)

search_config = BayesOptConfig(
    search_space= Cnn1DSearchSpace(),  # CnnSearchSpace(), # MlpSearchSpace(),  #  
    starting_points=10,
    checkpoint_dir="artifacts/cnn_test_dummy_dataset_low_boundaries"
)

bound_config = BoundConfig(
   error_bound=0.15,
   peak_mem_bound=20000,
   model_size_bound=50000,
   mac_bound=30000
)
