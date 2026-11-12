#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

OPSET = 17
rng = np.random.default_rng(0)


def fp16_exact(shape, scale=1.0, offset=0.0):
    return (rng.standard_normal(shape) * scale + offset).astype(np.float16).astype(np.float32)


def build(name, nodes, x, inits, dtype, y_shape=None):
    elem = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", elem, list(x.shape))],
        [helper.make_tensor_value_info("y", elem, y_shape)],
        [numpy_helper.from_array(v.astype(dtype), k) for k, v in inits.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    if y_shape is not None:
        onnx.checker.check_model(model)
    return model


def save(name, x, nodes, inits):
    model_fp32 = build(name, nodes, x, inits, np.float32)
    sess = ort.InferenceSession(model_fp32.SerializeToString(), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"x": x})[0]

    build(name, nodes, x, inits, np.float32, list(y.shape))
    model_fp16 = build(name, nodes, x, inits, np.float16, list(y.shape))
    onnx.save(model_fp16, os.path.join("models", name + ".onnx"))
    np.save(os.path.join("data", "input_" + name + ".npy"), x.astype(np.float16))
    np.save(os.path.join("data", "output_" + name + ".npy"), np.ascontiguousarray(y, dtype=np.float32))
    print(name, x.shape, "->", y.shape)


save("fp16_conv_relu_pools", fp16_exact((1, 8, 12, 12)), [
    helper.make_node("Conv", ["x", "W", "B"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c"], ["r"]),
    helper.make_node("AveragePool", ["r"], ["a"], kernel_shape=[3, 3], strides=[2, 2],
                     pads=[1, 1, 1, 1], count_include_pad=1),
    helper.make_node("MaxPool", ["a"], ["y"], kernel_shape=[2, 2], strides=[2, 2]),
], {"W": fp16_exact((16, 8, 3, 3), 0.1), "B": fp16_exact((16,), 0.1)})

save("fp16_gemm", fp16_exact((4, 32)), [
    helper.make_node("Gemm", ["x", "B", "C"], ["y"], alpha=0.5, beta=2.0, transB=0),
], {"B": fp16_exact((32, 24), 0.2), "C": fp16_exact((24,), 0.25)})

save("fp16_gemm_transB", fp16_exact((4, 32)), [
    helper.make_node("Gemm", ["x", "B", "C"], ["y"], transB=1),
], {"B": fp16_exact((24, 32), 0.2), "C": fp16_exact((4, 24), 0.25)})

save("fp16_matmul", fp16_exact((2, 3, 16)), [
    helper.make_node("MatMul", ["x", "B"], ["y"]),
], {"B": fp16_exact((16, 20), 0.25)})

save("fp16_layernorm_softmax", fp16_exact((2, 5, 32)), [
    helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["n"], axis=-1, epsilon=1e-5),
    helper.make_node("Softmax", ["n"], ["y"], axis=-1),
], {"scale": fp16_exact((32,), 0.1, 1.0), "bias": fp16_exact((32,), 0.1)})

save("fp16_add_sub_probe", fp16_exact((1, 64), 4.0), [
    helper.make_node("Add", ["x", "k"], ["t"]),
    helper.make_node("Sub", ["t", "k"], ["y"]),
], {"k": np.array([1024.0], dtype=np.float32)})
