from uNAS.config import PruningConfig
from configs.cnn_chars74k_aging import training_config, bound_config, search_config, search_algorithm

training_config.pruning = PruningConfig(
    structured=True,
    start_pruning_at_epoch=20,
    finish_pruning_by_epoch=53,
    min_sparsity=0.1,
    max_sparsity=0.85
)
