import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper

# Models for Fusion_MatMulToGemm.HigherRankAndColumnBias: MatMul + Add with a constant weight and bias, which
# the 5.x graph engine may rewrite into one Gemm. Models run by the test have data/input_<name>.npy and
# data/output_<name>.npy.


def save_case(name, x, w, bias, save_data=True):
    y = (x.astype(np.float64) @ w.astype(np.float64) + bias.astype(np.float64)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["MM"], name="matmul"),
            helper.make_node("Add", ["MM", "B"], ["Y"], name="add"),
        ],
        name,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(y.shape))],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(bias, "B")],
    )
    model = helper.make_model(
        graph,
        producer_name="matmul-add-bias-regression",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 8
    checker.check_model(model)
    onnx.save(model, "models/" + name + ".onnx")
    if save_data:
        np.save("data/input_" + name + ".npy", x)
        np.save("data/output_" + name + ".npy", y)


rng = np.random.default_rng(0)


def uniform(*shape):
    return rng.uniform(-1, 1, shape).astype(np.float32)


# [1, 1, N] bias on a 3-D input: one value per output column, fuses into a Gemm with a 1-D bias
save_case("matmul_add_bias_nd", uniform(2, 5, 16), uniform(16, 24), uniform(1, 1, 24))

# [M, 1] bias with M == N: one value per output row, must not be read as a per-column bias
column = uniform(16, 1)
column[0, 0] += 2
column[1, 0] -= 2
save_case("matmul_add_bias_column", uniform(16, 12), uniform(12, 16), column)

# [1, 1, N] bias on a 2-D input: the output gets 3 dimensions, which a Gemm does not produce.
# The test checks only that the rewrite skips it, so no data is saved.
save_case("matmul_add_bias_rank", uniform(3, 8), uniform(8, 6), uniform(1, 1, 6), save_data=False)
