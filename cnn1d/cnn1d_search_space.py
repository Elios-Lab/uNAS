from typing import List

from search_space import SearchSpace, ArchType, SchemaType


from .cnn1d_morphisms import produce_all_morphs
from .cnn1d_random_generators import random_arch_1d
from .cnn1d_schema import get_schema


class Cnn1DSearchSpace(SearchSpace):
    input_shape = None
    num_classes = None

    def __init__(self, dropout=0.0):
        self.dropout = dropout

    @property
    def schema(self) -> SchemaType:
        return get_schema()

    def random_architecture(self) -> ArchType:
        return random_arch_1d()

    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        return produce_all_morphs(arch)

    def to_keras_model(self, arch: ArchType, input_shape=None, num_classes=None, **kwargs):
        input_shape = input_shape or self.input_shape
        return arch.to_keras_model(input_shape=input_shape or self.input_shape,
                                   num_classes=num_classes or self.num_classes,
                                   dropout=self.dropout,
                                   **kwargs)
