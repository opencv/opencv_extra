import onnx
from onnx import helper, TensorProto

def make_size_model(model_path, name, input_shape):
    # input_shape: e.g. ['N'] for 1D dynamic, or [] for scalar
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape if input_shape else [])
    out = helper.make_tensor_value_info("y", TensorProto.INT64, [])
    node = helper.make_node("Size", ["x"], ["y"])
    graph = helper.make_graph([node], name, [inp], [out])
    opset = helper.make_operatorsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset], producer_name="size-tests")
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

make_size_model("test_size_1d_model.onnx",
                "test_size_1d", ["N"])

# True scalar input (rank-0, shape []).
make_size_model("test_size_0d_model.onnx",
                "test_size_0d", [])
print("Wrote models.")
