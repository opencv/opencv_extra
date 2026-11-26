import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper

# Models for Net_CudaGraphCoherence.SetParamInvalidatesCapturedGraph: the test replaces the weight "W"
# with Net::setParam("Y", 0, ...) before and after a CUDA Graph is recorded.


def save_model(node, inputs, outputs, initializers, path):
    graph = helper.make_graph([node], node.name, inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        producer_name="setparam-regression",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 8
    checker.check_model(model)
    onnx.save(model, path)


rng = np.random.default_rng(0)

# Y = X * W + B
M, K, N = 4, 8, 6
save_model(
    helper.make_node("Gemm", ["X", "W", "B"], ["Y"], name="gemm_setparam"),
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])],
    [
        numpy_helper.from_array(rng.uniform(-1, 1, (K, N)).astype(np.float32), "W"),
        numpy_helper.from_array(rng.uniform(-0.5, 0.5, (N,)).astype(np.float32), "B"),
    ],
    "models/gemm_setparam.onnx",
)

# 3x3 convolution without bias, 3 to 4 channels
save_model(
    helper.make_node("Conv", ["X", "W"], ["Y"], name="conv_setparam"),
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 6, 6])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4, 4])],
    [numpy_helper.from_array(rng.uniform(-0.5, 0.5, (4, 3, 3, 3)).astype(np.float32), "W")],
    "models/conv_setparam.onnx",
)
