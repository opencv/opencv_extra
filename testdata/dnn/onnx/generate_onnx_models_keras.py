from __future__ import print_function
import keras
import numpy as np
import os.path
import onnx
import google.protobuf.text_format
import io

import keras2onnx
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Layer
from keras import backend as K
import tensorflow as tf


def save_data_and_model(name, input_shape, model, inp=None, version=None):
    print(name + " input has sizes",  input_shape)
    input_files = os.path.join("data", "input_" + name)
    input = np.random.random(input_shape)
    np.save(input_files, input)

    output = model.predict(input)

    print(name + " output has sizes", output.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output))

    models_files = os.path.join("models", name + ".onnx")

    onnx_model = keras2onnx.convert_keras(model, name, target_opset=version, channel_first_inputs=[inp])
    onnx.save_model(onnx_model, models_files)

def save_onnx_data_and_model(input, output, name, operation, *args, **kwargs):
    print(name + " input has sizes",  input.shape)
    input_files = os.path.join("data", "input_" + name)
    input = input.astype(np.float32)
    np.save(input_files, input)

    print(name + " output has sizes", output.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    output = output.astype(np.float32)
    np.save(output_files, np.ascontiguousarray(K.eval(output)))

    models_files = os.path.join("models", name + ".onnx")
    X = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input.shape)
    Y = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, output.shape)
    node = onnx.helper.make_node(operation, inputs=['input'], outputs=['output'], *args, **kwargs)
    graph = onnx.helper.make_graph([node], name, [X], [Y])
    model = onnx.helper.make_model(graph, producer_name=name)
    onnx.save(model, models_files)


#keras2onnx.convert_keras(model, name=None, doc_string='', target_opset=None, channel_first_inputs=None):
    # type: (keras.Model, str, str, int, []) -> onnx.ModelProto
    """
    :param model: keras model
    :param name: the converted onnx model internal name
    :param doc_string:
    :param target_opset:
    :param channel_first_inputs: A list of channel first input.
    :return:
    """

class RoundLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RoundLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RoundLayer, self).build(input_shape)

    def call(self, x):
        return K.round(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

shape=[1,2,3,4]
a = Input(shape=shape[1:])
b = RoundLayer(shape[1:])(a)
model = Model(inputs=a, outputs=b)
save_data_and_model("round", shape, model, a, 11)

class LogLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(LogLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LogLayer, self).build(input_shape)

    def call(self, x):
        return K.log(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

shape=[1,2,3,4]
a = Input(shape=shape[1:])
b = LogLayer(shape[1:])(a)
model = Model(inputs=a, outputs=b)
save_data_and_model("log", shape, model, a, 11)

class SqrtLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SqrtLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SqrtLayer, self).build(input_shape)

    def call(self, x):
        return K.sqrt(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

shape=[1,2,3,4]
a = Input(shape=shape[1:])
b = SqrtLayer(shape[1:])(a)
model = Model(inputs=a, outputs=b)
save_data_and_model("sqrt", shape, model, a, 11)

class CeilLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CeilLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CeilLayer, self).build(input_shape)

    def call(self, x):
        return tf.math.ceil(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

shape=[1,2,3,4]
a = Input(shape=shape[1:])
b = CeilLayer(shape[1:])(a)
model = Model(inputs=a, outputs=b)
save_data_and_model("ceil", shape, model, a, 11)

class FloorLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FloorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FloorLayer, self).build(input_shape)

    def call(self, x):
        return tf.math.floor(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

shape=[1,2,3,4]
a = Input(shape=shape[1:])
b = FloorLayer(shape[1:])(a)
model = Model(inputs=a, outputs=b)
save_data_and_model("floor", shape, model, a, 11)

