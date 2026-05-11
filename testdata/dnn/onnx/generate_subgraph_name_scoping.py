import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def make_subgraph_name_scoping(out_path):
    # Outer Constant uses the name "shared" as its output.
    outer_shared = helper.make_node(
        "Constant", inputs=[], outputs=["shared"], name="outer_shared",
        value=numpy_helper.from_array(np.array([10.0, 20.0], dtype=np.float32)),
    )
    # Real outer use of "shared" so simplification cannot fold it away.
    add_outer = helper.make_node(
        "Add", inputs=["x", "shared"], outputs=["sum_outer"], name="add_outer",
    )

    # then_branch: a Constant whose output name ("shared") collides with the
    # outer scope. ONNX permits this; OpenCV must rescope it to a body-local arg.
    then_shared = helper.make_node(
        "Constant", inputs=[], outputs=["shared"], name="then_shared",
        value=numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32)),
    )
    then_id = helper.make_node("Identity", ["shared"], ["then_out"], name="then_id")
    then_graph = helper.make_graph(
        nodes=[then_shared, then_id], name="then_branch", inputs=[],
        outputs=[helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )

    # else_branch: same collision pattern with a different value.
    else_shared = helper.make_node(
        "Constant", inputs=[], outputs=["shared"], name="else_shared",
        value=numpy_helper.from_array(np.array([100.0, 200.0], dtype=np.float32)),
    )
    else_id = helper.make_node("Identity", ["shared"], ["else_out"], name="else_id")
    else_graph = helper.make_graph(
        nodes=[else_shared, else_id], name="else_branch", inputs=[],
        outputs=[helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )

    if_node = helper.make_node(
        "If", inputs=["cond"], outputs=["branch_val"], name="if_node",
        then_branch=then_graph, else_branch=else_graph,
    )

    graph = helper.make_graph(
        nodes=[outer_shared, add_outer, if_node],
        name="subgraph_name_scoping",
        inputs=[
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        outputs=[
            helper.make_tensor_value_info("sum_outer", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("branch_val", TensorProto.FLOAT, [2]),
        ],
    )

    opset = helper.make_operatorsetid("", 19)
    model = helper.make_model(
        graph, opset_imports=[opset], producer_name="subgraph-scoping-test",
    )
    # Note: onnx.checker rejects this model under strict SSA, but real-world
    # ONNX models exported by PyTorch/TF do produce subgraph-local names that
    # shadow the enclosing scope. That is precisely the case OpenCV's importer
    # must handle, so we deliberately skip onnx.checker here.
    onnx.save(model, out_path)


if __name__ == "__main__":
    make_subgraph_name_scoping("models/subgraph_name_scoping.onnx")
    print("Wrote models/subgraph_name_scoping.onnx")
