#!/usr/bin/env python3
"""Generate small ONNX importer compatibility fixtures.

Requires numpy and onnx. Run from any directory; models are written beside this
script in models/. numpy_helper serializes raw tensor data as little endian.
The variable QGemm C and unsupported deconvolution padding fixtures are valid
models whose configurations are explicitly rejected by OpenCV.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


MODELS = Path(__file__).resolve().parent / "models"


def tensor(name, values, dtype):
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name)


def value(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def save(name, node, inputs, output, initializers, check=True):
    graph = helper.make_graph([node], name, inputs, [output], initializers)
    model = helper.make_model(
        graph, producer_name="opencv_extra",
        opset_imports=[helper.make_opsetid("", 13),
                       helper.make_opsetid("com.microsoft", 1)], ir_version=7,
    )
    if check:
        onnx.checker.check_model(model)
    MODELS.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, MODELS / (name + ".onnx"))


def quantization(prefix):
    return [tensor(prefix + "_scale", [0.5], np.float32),
            tensor(prefix + "_zp", [0], np.int8)]


def qconv(name="importer_qconv_bias", bias_values=(10, -10), bias_dtype=np.int32, check=True):
    initializers = quantization("x") + quantization("w") + quantization("y")
    initializers += [tensor("w", np.ones((2, 1, 1, 1)), np.int8),
                     tensor("b", list(bias_values), bias_dtype)]
    node = helper.make_node(
        "QLinearConv", ["x", "x_scale", "x_zp", "w", "w_scale", "w_zp",
                        "y_scale", "y_zp", "b"], ["y"],
        kernel_shape=[1, 1], strides=[1, 1],
    )
    save(name, node, [value("x", TensorProto.INT8, [1, 1, 2, 2])],
         value("y", TensorProto.INT8, [1, 2, 2, 2]), initializers, check=check)


def qgemm(bias, output_zero_point=True, name=None, bias_values=(5, -5), bias_dtype=np.int32, check=True):
    initializers = quantization("a") + quantization("b") + quantization("y")
    initializers.append(tensor("b", [[1, 1], [1, 1]], np.int8))
    inputs = [value("a", TensorProto.INT8, [1, 2])]
    if bias == "constant":
        initializers.append(tensor("c", list(bias_values), bias_dtype))
    elif bias == "variable":
        inputs.append(value("c", TensorProto.INT32, [2]))
    node_inputs = ["a", "a_scale", "a_zp", "b", "b_scale", "b_zp",
                   "" if bias == "omitted" else "c", "y_scale"]
    if output_zero_point:
        node_inputs.append("y_zp")
    else:
        initializers = [t for t in initializers if t.name != "y_zp"]
    node = helper.make_node("QGemm", node_inputs, ["y"], domain="com.microsoft")
    suffix = "" if output_zero_point else "_no_output_zp"
    out_name = name if name is not None else ("importer_qgemm_" + bias + suffix)
    save(out_name, node, inputs,
         value("y", TensorProto.INT8, [1, 2]), initializers, check=check)


def lstm(name="importer_lstm_rank3", x_shape=(2, 1, 4), check=True):
    initializers = [tensor("W", np.full((1, 12, 4), 0.1), np.float32),
                    tensor("R", np.full((1, 12, 3), 0.1), np.float32),
                    tensor("B", np.zeros((1, 24)), np.float32)]
    node = helper.make_node("LSTM", ["X", "W", "R", "B"], ["Y"], hidden_size=3)
    save(name, node, [value("X", TensorProto.FLOAT, list(x_shape))],
         value("Y", TensorProto.FLOAT, [2, 1, 1, 3]), initializers, check=check)


def deconv(name, output_shape, kernel_shape=(3, 3), check=True, **attributes):
    node = helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                            kernel_shape=list(kernel_shape), output_shape=output_shape,
                            **attributes)
    w_shape = [1, 1] + list(kernel_shape)
    save("importer_deconv_" + name, node,
         [value("X", TensorProto.FLOAT, [1, 1, 2, 2])],
         value("Y", TensorProto.FLOAT, [1, 1] + output_shape[-2:]),
         [tensor("W", np.full(w_shape, 0.5), np.float32)], check=check)


def main():
    qconv()
    qconv("importer_qconv_float_bias", bias_values=(1.0, 2.0), bias_dtype=np.float32, check=False)
    qconv("importer_qconv_short_bias", bias_values=(10,), bias_dtype=np.int32, check=False)
    qgemm("constant")
    qgemm("omitted")
    qgemm("omitted", output_zero_point=False)
    qgemm("variable")
    qgemm("constant", name="importer_qgemm_float_bias", bias_values=(1.0, 2.0), bias_dtype=np.float32, check=False)
    qgemm("constant", name="importer_qgemm_short_bias", bias_values=(5,), bias_dtype=np.int32, check=False)
    lstm()
    lstm("importer_lstm_rank2", x_shape=(2, 4), check=False)
    lstm("importer_lstm_rank1", x_shape=(4,), check=False)
    deconv("same", [4, 4], strides=[2, 2], auto_pad="SAME_UPPER")
    # Retain support for the older full-tensor output_shape convention.
    deconv("legacy", [1, 1, 4, 4], strides=[2, 2], auto_pad="SAME_UPPER")
    deconv("no_auto_pad", [4, 4], strides=[2, 2])
    deconv("notset", [4, 4], strides=[2, 2], auto_pad="NOTSET")
    deconv("override_pads", [4, 4], strides=[2, 2], pads=[0, 0, 0, 0])
    deconv("default_stride", [2, 2])
    deconv("output_padding", [4, 4], strides=[2, 2], output_padding=[1, 1])
    deconv("unsupported_asymmetric", [3, 3])
    deconv("kernel_stride_mismatch", [4, 4], kernel_shape=(3,), strides=[2, 2], auto_pad="SAME_UPPER", check=False)
    deconv("output_shape_rank3", [1, 4, 4], strides=[2, 2], auto_pad="SAME_UPPER", check=False)
    deconv("zero_stride_same", [4, 4], strides=[0, 0], auto_pad="SAME_UPPER", check=False)
    deconv("zero_stride_notset", [4, 4], strides=[0, 0], check=False)
    deconv("negative_stride", [4, 4], strides=[-1, 2], auto_pad="SAME_UPPER", check=False)


if __name__ == "__main__":
    main()
