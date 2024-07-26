from typing import List

from uNAS.search_space import SearchSpace, SchemaType, ArchType
from .mlp_random_generators import random_arch_2d
from .mlp_morphisms import produce_all_morphs
from .mlp_schema import get_schema


class MlpSearchSpace(SearchSpace):
    """
    Multi-Layer Perceptron (MLP) search space for classification.
    ===============================================================================

    This module contains the search space for MLPs for classification.
    The search space is defined by the MlpSearchSpace class, which inherits from the SearchSpace class.

    The search space is travelled by the search algorithms to find the best architecture for the dataset.

    Random morphisms are applied to the architectures to generate new architectures. The morphisms are defined in the
    mlp_morphisms module.

    The architectures are converted to Keras models using the to_keras_model method.

    The search space is defined by the schema, which is defined in the mlp_schema module.
    """
    input_shape = None
    num_classes = None

    @property
    def schema(self) -> SchemaType:
        return get_schema()

    def random_architecture(self) -> ArchType:
        return random_arch_2d()

    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        return produce_all_morphs(arch)
