import numpy as np

from uNAS.architecture import Architecture
from uNAS.resource_models.graph import Graph
from uNAS.resource_models.ops import Input, Dense

class MlpArchitecture(Architecture):
    def __init__(self, architecture):
        self.architecture = architecture

    def to_keras_model(self, input_shape, num_classes, l2_reg=0.001, **kwargs):
        from keras import Model
        from keras.layers import Input, Flatten, Dense
        from keras.regularizers import l2

        i = Input(shape=input_shape)
        x = Flatten()(i)
        for dense_layer in self.architecture:
            x = Dense(dense_layer["units"], activation="relu",
                      kernel_regularizer=l2(l2_reg))(x)
        x = Dense(1 if num_classes <= 2 else num_classes)(x)

        return Model(inputs=i, outputs=x)

    def to_resource_graph(self, input_shape, num_classes, element_type=np.uint8, batch_size=1,
                          **kwargs):


        g = Graph(element_type)
        with g.as_current():
            x = Input(shape=(batch_size,) + input_shape)
            for dense_layer in self.architecture:
                x = Dense(dense_layer["units"], preflatten_input=True, activation="relu")(x)
            x = Dense(1 if num_classes <= 2 else num_classes)(x)
            g.add_output(x)

        return g
