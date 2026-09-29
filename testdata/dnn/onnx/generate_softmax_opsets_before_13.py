# Generates the Softmax/LogSoftmax models whose opset_import is below 13.
#
# The ONNX spec of those opsets coerces the input to the 2D tensor
# [a_0 * ... * a_{axis-1}, a_axis * ... * a_{n-1}] and reduces over its second
# dimension, so axis=1 on a 2x3x4 input normalises 12 values per sample instead of
# the 3 values of that axis.  Softmax-13 (the first opset with an explicit axis)
# reduces along the axis alone, which is what a plain 2D input cannot tell apart.
#
# The expected outputs are computed from that definition, and cross-checked against
# onnxruntime below.
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info


def reference(x, axis, log):
    shape = x.shape
    axis = axis + len(shape) if axis < 0 else axis
    outer = int(np.prod(shape[:axis]))
    inner = int(np.prod(shape[axis:]))
    flat = x.reshape(outer, inner).astype(np.float64)
    shifted = flat - flat.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    values /= values.sum(axis=1, keepdims=True)
    if log:
        values = np.log(values)
    return values.reshape(shape).astype(np.float32)


def write_case(label, op, axis, input_data):
    output_data = reference(input_data, axis, op == "LogSoftmax")

    node = make_node(op, ["x"], ["y"], axis=axis)
    graph = make_graph(
        [node],
        label,
        [make_tensor_value_info("x", TensorProto.FLOAT, input_data.shape)],
        [make_tensor_value_info("y", TensorProto.FLOAT, output_data.shape)],
    )
    model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 11)],
                       producer_name="opencv_extra", ir_version=8)
    onnx.save_model(model, os.path.join("models", "%s.onnx" % label))

    np.save(os.path.join("data", "input_%s.npy" % label), input_data)
    np.save(os.path.join("data", "output_%s.npy" % label), output_data)

    got = ort.InferenceSession(os.path.join("models", "%s.onnx" % label),
                               providers=["CPUExecutionProvider"]).run(None, {"x": input_data})[0]
    assert np.allclose(got, output_data, atol=1e-5), "%s: onnxruntime disagrees" % label


rng = np.random.RandomState(1234)
base = (rng.rand(2, 3, 4).astype(np.float32) - 0.5) * 4.0

write_case("softmax_axis_1_opset11", "Softmax", 1, base)
write_case("log_softmax_axis_0_opset11", "LogSoftmax", 0, base)
