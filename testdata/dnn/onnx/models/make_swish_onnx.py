import onnx
from onnx import helper, TensorProto

# Input
X = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 5])
Y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 5])

# Swish node
node = helper.make_node(
    'Swish',
    inputs=['x'],
    outputs=['y']
)

# Graph
graph = helper.make_graph(
    [node],
    'swish_graph',
    [X],
    [Y]
)

# Model
model = helper.make_model(graph, producer_name='opencv_test')

onnx.save(model, 'swish.onnx')
print("swish.onnx created")
