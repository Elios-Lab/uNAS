from typing import List

from uNAS.search_space import SearchSpace, ArchType, SchemaType
from .cnn_morphisms import produce_all_morphs
from .cnn_random_generators import random_arch
from .cnn_schema import get_schema


class CnnSearchSpace(SearchSpace):
    """
    2D Convolutional Neural Network (CNN) search space for image classification.
    ===============================================================================

    This module contains the search space for 2D CNNs for image classification.
    The search space is defined by the CnnSearchSpace class, which inherits from the SearchSpace class.

    The search space is travelled by the search algorithms to find the best architecture for the dataset.

    Random morphisms are applied to the architectures to generate new architectures. The morphisms are defined in the
    cnn_morphisms module.

    The architectures are converted to Keras models using the to_keras_model method.

    The search space is defined by the schema, which is defined in the cnn_schema module.
    """
    input_shape = None
    num_classes = None

    def __init__(self, dropout=0.0):
        self.dropout = dropout

    @property
    def schema(self) -> SchemaType:
        return get_schema()

    def random_architecture(self) -> ArchType:
        return random_arch()

    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        return produce_all_morphs(arch)

    def to_keras_model(self, arch: ArchType, input_shape=None, num_classes=None, **kwargs):
        input_shape = input_shape or self.input_shape
        return arch.to_keras_model(input_shape=input_shape or self.input_shape,
                                   num_classes=num_classes or self.num_classes,
                                   dropout=self.dropout,
                                   **kwargs)
