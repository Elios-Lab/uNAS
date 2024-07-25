from keras.models import Model
from abc import ABC, abstractmethod
from typing import List, Union, TypeVar, Dict
from uNAS.architecture import Architecture
from uNAS.resource_models.graph import Graph
from uNAS.schema_types import ValueType


ArchType = TypeVar('ArchType', bound=Architecture)
SchemaType = Dict[str, ValueType]


class SearchSpace(ABC):
    """
    Abstract class for search spaces.

    The search space is defined by the schema, which is a dictionary that defines the types of the hyperparameters
    of the architecture.

    The search space is travelled by the search algorithms to find the best architecture for the dataset.

    Random morphisms are applied to the architectures to generate new architectures.

    The architectures are converted to Keras models using the to_keras_model method.

    The search space is defined by the schema, which is defined in the schema module.

    The search space should implement the following methods:
    - schema: returns the schema of the search space
    - random_architecture: returns a random architecture from the search space
    - produce_morphs: produces morphs of the architecture
    - input_shape: returns the input shape of the architecture
    - num_classes: returns the number of classes in the dataset

    """
    @property
    @abstractmethod
    def schema(self) -> SchemaType:
        pass

    @abstractmethod
    def random_architecture(self) -> ArchType:
        pass

    @abstractmethod
    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        pass

    @property
    @abstractmethod
    def input_shape(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    def to_keras_model(self, arch: ArchType, input_shape=None, num_classes=None, **kwargs) -> Model:
        return arch.to_keras_model(input_shape=input_shape or self.input_shape,
                                   num_classes=num_classes or self.num_classes, **kwargs)

    def to_resource_graph(self, arch: ArchType,
                          input_shape=None, num_classes=None, **kwargs) -> Graph:
        return arch.to_resource_graph(input_shape=input_shape or self.input_shape,
                                      num_classes=num_classes or self.num_classes, **kwargs)
