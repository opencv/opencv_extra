import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper

# Models for Fusion_MatMulToGemm.HigherRankAndColumnBias: MatMul + Add with a constant weight and bias, which
# the 5.x graph engine may rewrite into one Gemm. Models run by the test have data/input_<name>.npy and
# data/output_<name>.npy.


def save_case(name, x, w, bias, save_data=True, pre_op=None):
    h = x
    nodes = []
    mm_in = "X"
    if pre_op == "Relu":
        h = np.maximum(x, 0)
        nodes.append(helper.make_node("Relu", ["X"], ["H"], name="relu"))
        mm_in = "H"
    elif pre_op == "Squeeze":
        h = np.squeeze(x)
        nodes.append(helper.make_node("Squeeze", ["X"], ["H"], name="squeeze"))
        mm_in = "H"
    elif pre_op is not None:
        raise ValueError(f"unsupported pre_op: {pre_op}")

    y = (h.astype(np.float64) @ w.astype(np.float64) + bias.astype(np.float64)).astype(np.float32)
    nodes.append(helper.make_node("MatMul", [mm_in, "W"], ["MM"], name="matmul"))
    nodes.append(helper.make_node("Add", ["MM", "B"], ["Y"], name="add"))
    graph = helper.make_graph(
        nodes,
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


def save_chain_case(name, x, w1, b1, w2, b2, save_data=True):
    h = (x.astype(np.float64) @ w1.astype(np.float64) + b1.astype(np.float64)).astype(np.float32)
    y = (h.astype(np.float64) @ w2.astype(np.float64) + b2.astype(np.float64)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W1"], ["MM1"], name="matmul1"),
            helper.make_node("Add", ["MM1", "B1"], ["H"], name="add1"),
            helper.make_node("MatMul", ["H", "W2"], ["MM2"], name="matmul2"),
            helper.make_node("Add", ["MM2", "B2"], ["Y"], name="add2"),
        ],
        name,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(y.shape))],
        initializer=[
            numpy_helper.from_array(w1, "W1"),
            numpy_helper.from_array(b1, "B1"),
            numpy_helper.from_array(w2, "W2"),
            numpy_helper.from_array(b2, "B2"),
        ],
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

# [1, 1, N] bias on a MatMul fed by Relu on a 3-D input: rank propagates across Relu and fuses into Gemm
save_case("matmul_add_bias_nd_after_relu", uniform(2, 5, 16), uniform(16, 24), uniform(1, 1, 24), pre_op="Relu")

# Two chained MatMul + Add([1, 1, N]) stages on a 3-D input: rank propagates across the first Gemm so both fuse
save_chain_case(
    "matmul_add_bias_nd_chain",
    uniform(2, 5, 16),
    uniform(16, 12),
    uniform(1, 1, 12),
    uniform(12, 24),
    uniform(1, 1, 24),
)

# 1-D [N] bias after data-dependent Squeeze (unknown output rank at fusion time): a 1-D bias never raises rank, so it fuses
save_case("matmul_add_bias_1d_rank_unknown", uniform(1, 3, 8), uniform(8, 6), uniform(6), pre_op="Squeeze")

# [1, 1, N] bias after Relu on a 2-D input, across two chained MatMul + Add stages on a 2-D input, and after
# data-dependent Squeeze (unknown rank at fusion time): all must skip the Gemm rewrite so no axis is lost.
save_case("matmul_add_bias_rank_after_relu", uniform(3, 8), uniform(8, 6), uniform(1, 1, 6), save_data=False, pre_op="Relu")
save_chain_case(
    "matmul_add_bias_rank_chain",
    uniform(3, 8),
    uniform(8, 6),
    uniform(1, 1, 6),
    uniform(6, 4),
    uniform(1, 1, 4),
    save_data=False,
)
save_case("matmul_add_bias_rank_unknown", uniform(1, 3, 8), uniform(8, 6), uniform(1, 1, 6), save_data=False, pre_op="Squeeze")
