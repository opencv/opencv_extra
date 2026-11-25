import numpy as np
import onnx
from onnx import helper, TensorProto

# DFT at opset 20 with no axis input. The spec default axis is -2.
name = "dft_opset20_default_axis"
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 2])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 2])
node = helper.make_node("DFT", ["x"], ["y"])
graph = helper.make_graph([node], name, [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 20)], producer_name=name)
onnx.checker.check_model(model)
onnx.save(model, "models/" + name + ".onnx")

# The last dim holds the real and imaginary parts. Transform along axis 2 (-2).
np.random.seed(0)
X = np.random.rand(1, 3, 4, 2).astype(np.float32)
Z = np.fft.fft(X[..., 0] + 1j * X[..., 1], axis=2)
Y = np.stack([Z.real, Z.imag], axis=-1).astype(np.float32)

np.save("data/input_" + name + ".npy", X)
np.save("data/output_" + name + ".npy", Y)
