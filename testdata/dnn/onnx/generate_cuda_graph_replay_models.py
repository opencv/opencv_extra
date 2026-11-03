# Models for the repeated-forward tests: CUDA records the second forward() as a CUDA Graph
# and replays it from the third, so every op in them must be safe to capture.
# Run from testdata/dnn/onnx: python generate_cuda_graph_replay_models.py
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def save(name, graph, inp, out):
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)],
                              producer_name="cuda-graph-replay-tests")
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join("models", name + ".onnx"))
    np.save(os.path.join("data", "input_" + name), inp)
    np.save(os.path.join("data", "output_" + name), np.ascontiguousarray(out))
    print(name, inp.shape, "->", out.shape)


rng = np.random.default_rng(0)

# B is broadcast over the leading batch axis, so the batch offsets of B are {0,s,2s,0,s,2s}:
# not evenly spaced, which is the case gemmBatched used to serve with device pointer arrays.
a = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
b = rng.standard_normal((3, 5, 6)).astype(np.float32)
graph = helper.make_graph(
    [helper.make_node("MatMul", ["A", "B"], ["Y"])], "matmul_batch_bcast",
    [helper.make_tensor_value_info("A", TensorProto.FLOAT, list(a.shape))],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4, 6])],
    [numpy_helper.from_array(b, "B")])
save("matmul_batch_bcast", graph, a, np.matmul(a, b))

# Shape feeds only static ops, so the whole graph is recorded with the Shape op inside it.
x = rng.standard_normal((1, 4, 8)).astype(np.float32)
graph = helper.make_graph(
    [helper.make_node("Shape", ["X"], ["shape"]),
     helper.make_node("Gather", ["shape", "index"], ["dim"], axis=0),
     helper.make_node("Cast", ["dim"], ["dim_f"], to=TensorProto.FLOAT),
     helper.make_node("Sqrt", ["dim_f"], ["scale"]),
     helper.make_node("Mul", ["X", "scale"], ["Y"])],
    "shape_scale_static",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x.shape))],
    [numpy_helper.from_array(np.array([2], dtype=np.int64), "index")])
save("shape_scale_static", graph, x, x * np.sqrt(np.float32(x.shape[2])))

# The batch is symbolic, so the Shape chain is not folded: it runs on the host and only sizes the Reshape.
x = rng.standard_normal((2, 4, 8)).astype(np.float32)
graph = helper.make_graph(
    [helper.make_node("Shape", ["X"], ["shape"]),
     helper.make_node("Gather", ["shape", "index"], ["n"], axis=0),
     helper.make_node("Unsqueeze", ["n", "axes"], ["n_1d"]),
     helper.make_node("Concat", ["n_1d", "tail"], ["new_shape"], axis=0),
     helper.make_node("Reshape", ["X", "new_shape"], ["flat"]),
     helper.make_node("Relu", ["flat"], ["Y"])],
    "shape_reshape_dynamic",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 4, 8])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 32])],
    [numpy_helper.from_array(np.array(0, dtype=np.int64), "index"),
     numpy_helper.from_array(np.array([0], dtype=np.int64), "axes"),
     numpy_helper.from_array(np.array([32], dtype=np.int64), "tail")])
save("shape_reshape_dynamic", graph, x, np.maximum(x.reshape(2, 32), 0))
