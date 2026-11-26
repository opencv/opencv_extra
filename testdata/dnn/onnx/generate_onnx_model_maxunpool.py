import numpy as np
import onnx
from onnx import helper, TensorProto

# MaxUnpool with a non-square kernel and stride. ONNX orders both as [H, W].
name = "maxunpool_non_square"
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 3])
indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1, 2, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 3])
node = helper.make_node("MaxUnpool", ["x", "indices"], ["y"], kernel_shape=[2, 1], strides=[2, 1])
graph = helper.make_graph([node], name, [x, indices], [y])
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)], producer_name=name)
onnx.checker.check_model(model)
onnx.save(model, "models/" + name + ".onnx")

# Each value came from one of the two rows of its 2x1 window in a 4x3 image.
# indices flatten that image: row * 3 + column.
X = np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32)
I = np.array([[[[0, 4, 2], [9, 7, 11]]]], dtype=np.int64)
Y = np.zeros((1, 1, 4, 3), dtype=np.float32)
Y.flat[I.ravel()] = X.ravel()

np.save("data/input_" + name + "_0.npy", X)
np.save("data/input_" + name + "_1.npy", I)
np.save("data/output_" + name + ".npy", Y)

# MaxUnpool with pads. ONNX orders 2D pads as [top, left, bottom, right].
name = "maxunpool_pads"
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1, 2, 2])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 4])
node = helper.make_node("MaxUnpool", ["x", "indices"], ["y"], kernel_shape=[2, 2], strides=[2, 2], pads=[1, 0, 1, 0])
graph = helper.make_graph([node], name, [x, indices], [y])
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)], producer_name=name)
onnx.checker.check_model(model)
onnx.save(model, "models/" + name + ".onnx")

# pads remove the top and bottom rows, so the output is 2x4.
# indices flatten that image: row * 4 + column.
X = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
I = np.array([[[[0, 3], [5, 6]]]], dtype=np.int64)
Y = np.zeros((1, 1, 2, 4), dtype=np.float32)
Y.flat[I.ravel()] = X.ravel()

np.save("data/input_" + name + "_0.npy", X)
np.save("data/input_" + name + "_1.npy", I)
np.save("data/output_" + name + ".npy", Y)