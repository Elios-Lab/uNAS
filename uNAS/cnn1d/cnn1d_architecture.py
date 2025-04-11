import numpy as np
import tensorflow as tf

from uNAS.architecture import Architecture
from uNAS.resource_models.graph import Graph
from uNAS.resource_models.ops import Conv1D, DWConv1D, Dense, Pool1D, Add, Input

class Cnn1DArchitecture(Architecture):
    def __init__(self, architecture_dict):
        self.architecture = architecture_dict

    def _assemble_a_network(self, input, num_classes, conv_layer,
                        pooling_layer, dense_layer, add_layer, flatten_layer):
        """
        Assembles a network architecture, starting at `input`, using factory functions for each
        layer type. Supports multi-channel input (T,C).
        :param input: Input tensor of shape (batch_size, timesteps, channels)
        :returns Output tensor of the network
        """

        def tie_up_pending_outputs(outputs):
            if len(outputs) == 1:
                return outputs[0]
            smallest_l = min(o.shape[1] for o in outputs)
            xs = []
            for o in outputs:
                downsampling_l = int(round(o.shape[1] / smallest_l))
                if downsampling_l > 1:
                    o = pooling_layer(o, {
                        "pool_size": downsampling_l,
                        "type": "max"
                    })
                xs.append(o)
            return add_layer(xs)

        li = None
        xs = [input]
        for conv_block in self.architecture["conv_blocks"]:
            if conv_block["is_branch"]:
                assert li is not None, "The first block can't be a branch block."
                previous_channels = xs[0].shape[-1]

                x = li
                _1x1 = False
                for j, l in enumerate(conv_block["layers"]):
                    if j == len(conv_block["layers"]) - 1:
                        if l["type"] == "DWConv1D":
                            _1x1 = True
                        else:
                            l["filters"] = previous_channels
                    x = conv_layer(x, l)

                if _1x1:
                    x = conv_layer(x, {
                        "type": "1x1Conv1D",
                        "filters": previous_channels,
                        "has_bn": False,
                        "has_relu": False,
                        "has_prepool": False,
                    })
                xs.append(x)

            else:
                x = tie_up_pending_outputs(xs)
                li = x
                for l in conv_block["layers"]:
                    x = conv_layer(x, l)
                xs = [x]

        x = tie_up_pending_outputs(xs)
        pooling = self.architecture["pooling"]
        if pooling:
            x = pooling_layer(x, pooling)

        x = flatten_layer(x)

        for l in self.architecture["dense_blocks"]:
            x = dense_layer(x, l)

        final_dense = self.architecture.get("_final_dense", {})
        final_dense.update({
            "units": 1 if num_classes == 2 else num_classes,
            "activation": None
        })
        x = dense_layer(x, final_dense)
        self.architecture["_final_dense"] = final_dense
        return x


    def to_keras_model(self, input_shape, num_classes, dropout=0.0, **kwargs):
        """
        Creates a Keras model for the candidate architecture.
        :param input_shape: Tuple of (timesteps, channels)
        """
        from keras import Model
        from keras.layers import Conv1D, DepthwiseConv1D, BatchNormalization, \
            ReLU, Dense, Input, Add, Flatten, MaxPool1D, AvgPool1D, ZeroPadding1D, Dropout, Reshape

        if len(input_shape) != 2:
            raise ValueError(f"Input shape must be (timesteps, channels), got {input_shape}")

        i = Input(shape=input_shape)

        def conv_layer(x, l):
            if l["has_prepool"] and x.shape[1] > 1:
                pool_size = min(2, x.shape[1])
                x = MaxPool1D(pool_size=pool_size)(x)

            kernel_size = 1 if l["type"] == "1x1Conv1D" else min(l["ker_size"], x.shape[1])
            stride = 1 if l["type"] == "1x1Conv1D" or not l["1x_stride"] else 2

            if l["type"] in ["Conv1D", "1x1Conv1D"]:
                x = Conv1D(filters=l["filters"],
                          kernel_size=kernel_size,
                          strides=stride,
                          padding="same")(x)
            else:
                x = DepthwiseConv1D(kernel_size=kernel_size,
                                   strides=stride,
                                   padding="same",
                                   depth_multiplier=1)(x)

            if l["has_bn"]:
                x = BatchNormalization()(x)
            if l["has_relu"]:
                x = ReLU()(x)
            return x

        def pooling_layer(x, l):
            pool_size = l["pool_size"] if isinstance(l["pool_size"], int) else l["pool_size"][0]
            pool_size = min(pool_size, x.shape[1])
            if l["type"] == "avg":
                x = AvgPool1D(pool_size=pool_size, padding="same")(x)
            else:
                x = MaxPool1D(pool_size=pool_size, padding="same")(x)
            return x

        def dense_layer(x, l):
            if len(x.shape) > 2:
                x = Flatten()(x)
            if dropout > 0.0:
                x = Dropout(dropout)(x)
            x = Dense(units=l["units"], activation=l["activation"])(x)
            return x

        def add_layer(xs):
            max_length = max(x.shape[1] for x in xs)
            os = []
            for x in xs:
                l_diff = max_length - x.shape[1]
                if l_diff > 0:
                    x = ZeroPadding1D(padding=(0, l_diff))(x)
                os.append(x)
            result = Add()(os)
            return result

        def flatten_layer(x):
            if len(x.shape) > 2:
                x = Flatten()(x)
            return x

        o = self._assemble_a_network(i, num_classes, conv_layer, pooling_layer, dense_layer, add_layer, flatten_layer)
        model = Model(inputs=i, outputs=o)
        
        model.summary()
        return model

    def to_resource_graph(self, input_shape, num_classes, element_type=np.uint8, batch_size=1,
                          pruned_weights=None):
        """
        Assembles a resource graph for the model, which can be used to compute runtime properties.
        """


        pruned_weights = {w.name: w for w in pruned_weights} if pruned_weights else {}

        def process_pruned_weights(l):
            """ If pruned weights are available, this will extract the correct number of
                channels / units from the weight matrix, and the number of non-zero entries from
                the surviving channels / units. """
            if "_weights" not in l:
                return None, None

            name = l["_weights"][0]  # Kernel matrix is the first entry
            if name not in pruned_weights:
                return None, None

            w = pruned_weights[name]
            channel_counts = tf.math.count_nonzero(tf.reshape(w, (-1, w.shape[-1])), axis=0)
            units = int(tf.math.count_nonzero(channel_counts).numpy())
            sparse_size = sum(channel_counts.numpy())
            return max(units, 1), sparse_size

        def conv_layer(x, l):
            if l["has_prepool"] and x.shape[1] > 1:
                pool_size = min(2, x.shape[1])
                x = Pool1D(type="max", pool_size=pool_size)(x)

            kernel_size = 1 if l["type"] == "1x1Conv1D" else min(l["ker_size"], x.shape[1])
            stride = 1 if l["type"] == "1x1Conv1D" or not l["1x_stride"] else 2
            if l["type"] in ["Conv1D", "1x1Conv1D"]:
                filters, sparse_kernel_length = process_pruned_weights(l)
                x = Conv1D(filters=filters or l["filters"],
                           kernel_size=kernel_size, stride=stride, padding="valid",
                           batch_norm=l["has_bn"], activation="relu" if l["has_relu"] else None,
                           sparse_kernel_size=sparse_kernel_length)(x)
            else:
                assert l["type"] == "DWConv1D"
                _, sparse_kernel_length = process_pruned_weights(l)
                x = DWConv1D(kernel_size=kernel_size, stride=stride,
                             padding="valid", batch_norm=l["has_bn"],
                             activation="relu" if l["has_relu"] else None,
                             sparse_kernel_size=sparse_kernel_length)(x)
            return x

        def pooling_layer(x, l):
            pool_size = l["pool_size"] if isinstance(l["pool_size"], int) else l["pool_size"][0]
            pool_size = min(pool_size, x.shape[1])
            return Pool1D(pool_size=pool_size, type=l["type"])(x)

        def dense_layer(x, l):
            units, sparse_kernel_size = process_pruned_weights(l)
            return Dense(units=units or l["units"],
                         preflatten_input=True, activation=l["activation"],
                         sparse_kernel_size=sparse_kernel_size)(x)

        def add_layer(xs):
            return Add(all_equal_shape=False)(xs)

        def flatten_layer(x):
            return x

        g = Graph(element_type)
        with g.as_current():
            i = Input(shape=(batch_size,) + input_shape)
            o = self._assemble_a_network(i, num_classes, conv_layer, pooling_layer, dense_layer, add_layer, flatten_layer)
            g.add_output(o)

        return g
