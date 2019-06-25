# This script is used to generate test data for OpenCV deep learning module.
import numpy as np
import tensorflow as tf
import os
import argparse
import struct
import cv2 as cv

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

np.random.seed(2701)

def gen_data(placeholder):
    shape = placeholder.shape.as_list()
    shape[0] = shape[0] if shape[0] else 1  # batch size = 1 instead None
    return np.random.standard_normal(shape).astype(placeholder.dtype.as_numpy_dtype())

def prepare_for_dnn(sess, graph_def, in_node, out_node, out_graph, dtype, optimize=True, quantize=False):
    # Freeze graph. Replaces variables to constants.
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [out_node])
    if optimize:
        # Optimize graph. Removes training-only ops, unused nodes.
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, [in_node], [out_node], dtype.as_datatype_enum)
        # Fuse constant operations.
        transforms = ["fold_constants(ignore_errors=True)"]
        if quantize:
            transforms += ["quantize_weights(minimum_size=0)"]
        transforms += ["sort_by_execution_order"]
        graph_def = TransformGraph(graph_def, [in_node], [out_node], transforms)
    # Serialize
    with tf.gfile.FastGFile(out_graph, 'wb') as f:
        f.write(graph_def.SerializeToString())

tf.reset_default_graph()
tf.Graph().as_default()
tf.set_random_seed(324)
sess = tf.Session()

# Use this variable to switch behavior of layers.
isTraining = tf.placeholder(tf.bool, name='isTraining')

def writeBlob(data, name):
    if data.ndim == 4:
        # NHWC->NCHW
        np.save(name + '.npy', data.transpose(0, 3, 1, 2).astype(np.float32))
    elif data.ndim == 5:
        # NDHWC->NCDHW
        np.save(name + '.npy', data.transpose(0, 4, 1, 2, 3).astype(np.float32))
    else:
        # Save raw data.
        np.save(name + '.npy', data.astype(np.float32))

def runModel(inpName, outName, name):
    with tf.Session(graph=tf.Graph()) as localSession:
        localSession.graph.as_default()

        with tf.gfile.FastGFile(name + '_net.pb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        inputData = gen_data(inp)
        outputData = localSession.run(localSession.graph.get_tensor_by_name(outName),
                                      feed_dict={localSession.graph.get_tensor_by_name(inp.name): inputData})
        writeBlob(inputData, name + '_in')
        writeBlob(outputData, name + '_out')

def save(inp, out, name, quantize=False, optimize=True):
    sess.run(tf.global_variables_initializer())

    inputData = gen_data(inp)
    outputData = sess.run(out, feed_dict={inp: inputData, isTraining: False})
    writeBlob(inputData, name + '_in')
    writeBlob(outputData, name + '_out')

    prepare_for_dnn(sess, sess.graph.as_graph_def(), inp.name[:inp.name.rfind(':')],
                    out.name[:out.name.rfind(':')], name + '_net.pb', inp.dtype,
                    optimize, quantize)

    # By default, float16 weights are stored in repeated tensor's field called
    # `half_val`. It has type int32 with leading zeros for unused bytes.
    # This type is encoded by Varint that means only 7 bits are used for value
    # representation but the last one is indicated the end of encoding. This way
    # float16 might takes 1 or 2 or 3 bytes depends on value. To impove compression,
    # we replace all `half_val` values to `tensor_content` using only 2 bytes for everyone.
    with tf.gfile.FastGFile(name + '_net.pb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            if 'value' in node.attr:
                halfs = node.attr["value"].tensor.half_val
                if not node.attr["value"].tensor.tensor_content and halfs:
                    node.attr["value"].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)
                    node.attr["value"].tensor.ClearField('half_val')
        tf.train.write_graph(graph_def, "", name + '_net.pb', as_text=False)

# Test cases ###################################################################
# shape: NHWC

from tensorflow.python.ops.nn_grad import _MaxPoolGrad as MaxUnPooling2D

inp = tf.placeholder(tf.float32, [1, 7, 7, 3], 'input')
pool = tf.layers.max_pooling2d(inp, pool_size=(2, 2), strides=(2, 2))
conv = tf.layers.conv2d(inputs=pool, filters=3, kernel_size=[1, 1], padding='VALID')
unpool = MaxUnPooling2D(pool.op, conv)
save(inp, unpool, 'max_pool_grad')

# Uncomment to print the final graph.
# with tf.gfile.FastGFile('fused_batch_norm_net.pb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     print graph_def
