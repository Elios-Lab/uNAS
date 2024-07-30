import tensorflow as tf

from typing import List, Callable, Optional
from dataclasses import dataclass
from uNAS.search_space import SearchSpace
from dataset import Dataset


@dataclass
class DistillationConfig:
    """
    DistillationConfig
    ---------------

    Configuration for knowledge distillation from a teacher model to a student model.

    The distillation loss is computed as the Kullback-Leibler divergence between the teacher's logits and the student's logits.

    The distillation loss is added to the student's loss with a weight factor `alpha`.

    The teacher's logits are softened by dividing them by a temperature factor `temperature`.

    The distillation loss is computed as follows:
        D_KL(softmax(teacher_logits / temperature), softmax(student_logits / temperature))

    The distillation loss is added to the student's loss with a weight factor `alpha`.

    The teacher model is loaded from the `distill_from` path.

    Args:
    - distill_from: str, Path to a tf.keras.Model.
    - alpha: float, optional, Weight factor for the distillation loss (D_KL between teacher and student).
    - temperature: float, optional, Softening factor for teacher's logits.
    """
    distill_from: str  # a Path to a tf.keras.Model
    alpha: float = 0.3  # Weight factor for the distillation loss (D_KL between teacher and student)
    temperature: float = 1.0  # Softening factor for teacher's logits


@dataclass
class PruningConfig:
    """
    PruningConfig
    ---------------

    Configuration for structured pruning.
    
    The pruning is applied to the model's weights during training.

    Args:
    - structured: bool, optional, Whether to produce a sparse model (structured = False, default) or prune the model channel-wise (structured = True).
    - start_pruning_at_epoch: int, optional, Epoch to start pruning. Default is 0.
    - finish_pruning_by_epoch: int, optional, Epoch to finish pruning. Default is None.
    - min_sparsity: float, optional, Minimum sparsity level, computed as the fraction of values (or channels, if structured = True) set to 0. Default is 0.
    - max_sparsity: float, optional, Maximum sparsity level, computed as the fraction of values (or channels, if structured = True) set to 0. Default is 0.995.ù

    """
    structured: bool = False
    start_pruning_at_epoch: int = 0
    finish_pruning_by_epoch: int = None
    min_sparsity: float = 0
    max_sparsity: float = 0.995


@dataclass
class TrainingConfig:
    """
    TrainingConfig
    ---------------
    
    Configuration for training a model.

    Args:
    - dataset: Dataset
    - optimizer: Callable[[], tf.optimizers.Optimizer]
    - callbacks: Callable[[], List[tf.keras.callbacks.Callback]]
    - batch_size: int, optional, Default is 128.
    - epochs: int, optional, Default is 75.
    - distillation: Optional[DistillationConfig], optional, Default is None. No distillation if `None`.
    - use_class_weight: bool, optional, Default is False. Compute and use class weights to re-balance the data.
    - pruning: Optional[PruningConfig], optional, Default is None.
    """
    dataset: Dataset
    optimizer: Callable[[], tf.optimizers.Optimizer]
    callbacks: Callable[[], List[tf.keras.callbacks.Callback]]
    batch_size: int = 128
    epochs: int = 75
    distillation: Optional[DistillationConfig] = None  # No distillation if `None`
    use_class_weight: bool = False  # Compute and use class weights to re-balance the data
    pruning: Optional[PruningConfig] = None


@dataclass
class BayesOptConfig:
    """
    BayesOptConfig
    ---------------
    
    Configuration for Bayesian Optimisation.
    
    Args:
    - search_space: SearchSpace
    - multifidelity: bool, optional, Enables multi-fidelity optimisation for the accuracy/error model. Default is False.
        discard areas of the search space with low accuracy without taking other objectives into account.
    - starting_points: int, optional, Number of starting points for the optimisation. Default is 15.
    - rounds: int, optional, Number of rounds for the optimisation. Default is 800.
    - checkpoint_dir: str, optional, Path to the directory to store the checkpoints. Default is "artifacts".
    """

    search_space: SearchSpace
    # Enables multi-fidelity optimisation for the accuracy/error model. Note that this can
    # discard areas of the search space with low accuracy without taking other objectives into account.
    multifidelity: bool = False
    starting_points: int = 15
    rounds: int = 800
    checkpoint_dir: str = "artifacts"


@dataclass
class AgingEvoConfig:
    """
    AgingEvoConfig
    ---------------
    
    Configuration for Aging Evolution.

    Args:
    - search_space: SearchSpace
    - population_size: int, optional, Default is 100.
    - sample_size: int, optional, Default is 25.
    - initial_population_size: Optional[int], optional, Default is None. If None, equal to population_size.
    - rounds: int, optional, Default is 2000.
    - max_parallel_evaluations: Optional[int], optional, Default is None.
    - checkpoint_dir: str, optional, Default is "artifacts".
    """
    search_space: SearchSpace
    population_size: int = 100
    sample_size: int = 25
    initial_population_size: Optional[int] = None 
    rounds: int = 2000
    max_parallel_evaluations: Optional[int] = None
    checkpoint_dir: str = "artifacts"


@dataclass
class BoundConfig:
    """
    BoundConfig
    ---------------
    
    Configuration for the constraints on the models.

    NAS will attempt to find models whose metrics are below the specified bounds.

    None means that the constraint is not considered.
    
    Args:
    - error_bound: Optional[float], optional, Default is None.
    - peak_mem_bound: Optional[int], optional, Default is None.
    - model_size_bound: Optional[int], optional, Default is None.
    - mac_bound: Optional[int], optional, Default is None.
    """

    error_bound: Optional[float] = None
    peak_mem_bound: Optional[int] = None
    model_size_bound: Optional[int] = None
    mac_bound: Optional[int] = None


@dataclass
class ModelSaverConfig:
    """
    ModelSaverConfig
    ---------------
    
    Configuration for saving the models.
    
    Args:
    - save_criteria: Optional[str], optional, Options are "all", "boundaries", "pareto", "none". Default is None.
    """

    # The criteria to save the model. Options are "all", "boundaries", "pareto", "none".
    save_criteria: Optional[str] = None
