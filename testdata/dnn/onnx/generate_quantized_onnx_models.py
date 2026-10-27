from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import onnx # version >= 1.12.0
import onnxruntime as rt
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

class DataReader(CalibrationDataReader):
    def __init__(self, model_path, batchsize=5):
        sess = rt.InferenceSession(model_path, None)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        calib_data = np.random.uniform(-1, 1, size=[batchsize] + input_shape[1:]).astype("float32")
        self.enum_data_dicts = iter([{input_name: np.expand_dims(x, axis=0)} for x in calib_data])

    def get_next(self):
        return next(self.enum_data_dicts, None)

def quantize_and_save_model(name, input, model, act_type="uint8", wt_type="uint8", per_channel=False, ops_version = 13, quanFormat=QuantFormat.QOperator):
    float_model_path = os.path.join("models", "dummy.onnx")
    quantized_model_path = os.path.join("models", name + ".onnx")
    type_dict = {"uint8" : QuantType.QUInt8, "int8" : QuantType.QInt8}

    model.eval()
    torch.onnx.export(model, input, float_model_path, export_params=True, opset_version=ops_version)

    dr = DataReader(float_model_path)
    quantize_static(float_model_path, quantized_model_path, dr, quant_format=quanFormat, per_channel=per_channel,
                    activation_type=type_dict[act_type], weight_type=type_dict[wt_type])

    os.remove(float_model_path)
    # os.remove(os.path.join("models", "dummy-opt.onnx"))
    # os.remove("augmented_model.onnx")
    
    sess = rt.InferenceSession(quantized_model_path, None)
    input = np.random.uniform(-1, 1, sess.get_inputs()[0].shape).astype("float32")
    output = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name : input})[0]

    print(name + " input has sizes",  input.shape)
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)

    print(name + " output has sizes", output.shape)
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

torch.manual_seed(0)
np.random.seed(0)

input = Variable(torch.randn(1, 3, 10, 10))
conv = nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
# generate QOperator qunatized model
quantize_and_save_model("quantized_conv_uint8_weights", input, conv)
quantize_and_save_model("quantized_conv_int8_weights", input, conv, wt_type="int8")
quantize_and_save_model("quantized_conv_per_channel_weights", input, conv, per_channel=True)

# generate QDQ qunatized model
quantize_and_save_model("quantized_conv_uint8_weights_qdq", input, conv, quanFormat=QuantFormat.QDQ)
quantize_and_save_model("quantized_conv_int8_weights_qdq", input, conv, wt_type="int8", quanFormat=QuantFormat.QDQ)
quantize_and_save_model("quantized_conv_per_channel_weights_qdq", input, conv, per_channel=True, quanFormat=QuantFormat.QDQ)

input = Variable(torch.randn(1, 3))
linear = nn.Linear(3, 4, bias=True)
quantize_and_save_model("quantized_matmul_uint8_weights", input, linear)
quantize_and_save_model("quantized_matmul_int8_weights", input, linear, wt_type="int8")
quantize_and_save_model("quantized_matmul_per_channel_weights", input, linear, wt_type="int8", per_channel=True)

class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x):
      y = torch.t(x)
      return x @ y

model = MatMul()
input = Variable(torch.randn(1, 4))
quantize_and_save_model("quantized_matmul_variable_inputs", input, model, wt_type="int8")

class Eltwise(nn.Module):

    def __init__(self):
        super(Eltwise, self).__init__()
        self.squeeze1 = nn.Linear(2, 2, bias=False)
        self.squeeze2 = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        y = self.squeeze1(x)
        z = self.squeeze2(x)
        return z + x * y

input = Variable(torch.randn(1, 2))
model = Eltwise()
quantize_and_save_model("quantized_eltwise", input, model, wt_type="int8")

class EltwiseScalars(nn.Module):

    def __init__(self):
        super(EltwiseScalars, self).__init__()

    def forward(self, x):
        return -1.5 + 2.5 * x

input = Variable(torch.randn(1, 2, 3, 4))
model = EltwiseScalars()
quantize_and_save_model("quantized_eltwise_scalar", input, model)

class EltwiseBroadcast(nn.Module):

    def __init__(self):
        super(EltwiseBroadcast, self).__init__()
        self.a = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
        self.b = torch.tensor([[-0.9, 0.8]], dtype=torch.float32)

    def forward(self, x):
        return self.a[:, :, None, None] + self.b[:, :, None, None] * x

input = Variable(torch.randn(1, 2, 3, 4))
model = EltwiseBroadcast()
quantize_and_save_model("quantized_eltwise_broadcast", input, model)

input = Variable(torch.randn(1, 2, 3, 4))
leaky_relu = nn.LeakyReLU(negative_slope=0.25)
quantize_and_save_model("quantized_leaky_relu", input, leaky_relu)

input = Variable(torch.randn(1, 10))
sigmoid = nn.Sigmoid()
quantize_and_save_model("quantized_sigmoid", input, sigmoid)

input = Variable(torch.randn(1, 3, 6, 6))
max_pool = nn.Sequential(nn.Sigmoid(),
                         nn.MaxPool2d(kernel_size=2, stride=2))
quantize_and_save_model("quantized_maxpool", input, max_pool)

input = Variable(torch.randn(1, 3, 6, 6))
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
quantize_and_save_model("quantized_avgpool", input, avg_pool)

class SplitMax(nn.Module):

    def __init__(self):
        super(SplitMax, self).__init__()

    def forward(self, x):
        first, second = torch.split(x, (2, 4), dim=1)
        second, third = torch.split(second, (2, 2), dim=1)
        return torch.max(first, torch.max(second, third))

model = SplitMax()
input = Variable(torch.randn(1, 6, 2, 3))
quantize_and_save_model("quantized_split", input, model)

input = Variable(torch.randn(1, 3, 4, 5))
model = nn.Sequential(nn.LeakyReLU(negative_slope=0.1),
                      nn.ZeroPad2d((2, 1, 2, 1)),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReflectionPad2d(1))
quantize_and_save_model("quantized_padding", input, model)

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        return torch.reshape(self.relu(x), (1, -1))

input = Variable(torch.randn(1, 3, 4, 4))
model = Reshape()
quantize_and_save_model("quantized_reshape", input, model)

class Transpose(nn.Module):

    def __init__(self):
        super(Transpose, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return torch.t(self.relu(x))

input = Variable(torch.randn(1, 3))
model = Transpose()
quantize_and_save_model("quantized_transpose", input, model)

class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return torch.squeeze(self.sigmoid(x), dim=2)

input = Variable(torch.randn(1, 3, 1, 4))
model = Squeeze()
quantize_and_save_model("quantized_squeeze", input, model)

class Unsqueeze(nn.Module):

    def __init__(self):
        super(Unsqueeze, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return torch.unsqueeze(self.sigmoid(x), dim=1)

input = Variable(torch.randn(1, 2, 3))
model = Unsqueeze()
quantize_and_save_model("quantized_unsqueeze", input, model)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Sequential(nn.LeakyReLU(negative_slope=0.1),
                       nn.Upsample(scale_factor=2, mode='nearest'))
quantize_and_save_model("quantized_resize_nearest", input, resize)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Sequential(nn.Sigmoid(),
                       nn.Upsample(size=[6, 8], mode='bilinear'))
quantize_and_save_model("quantized_resize_bilinear", input, resize)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Sequential(nn.Sigmoid(),
                       nn.Upsample(size=[6, 8], mode='bilinear', align_corners=True))
quantize_and_save_model("quantized_resize_bilinear_align", input, resize)

class Concatenation(nn.Module):

    def __init__(self):
        super(Concatenation, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        y = self.relu(x)
        return torch.cat([x, y], axis=1)

input = Variable(torch.randn(1, 2, 2, 2))
model = Concatenation()
quantize_and_save_model("quantized_concat", input, model)

class ConcatConstBlob(nn.Module):

    def __init__(self):
        super(ConcatConstBlob, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        y = torch.tensor([[[[0.1, -0.2], [-0.3, 0.4]]]], dtype=torch.float32)
        return torch.cat([x, y], axis=1)

input = Variable(torch.randn(1, 2, 2, 2))
model = ConcatConstBlob()
quantize_and_save_model("quantized_concat_const_blob", input, model)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1, 16*5*5)

model = nn.Sequential(
                nn.Conv2d(3,6,5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
                )
input = Variable(torch.randn(1, 3, 32, 32))
quantize_and_save_model("quantized_constant", input, model, wt_type="int8", per_channel=True)

class Gemm(nn.Module):
    def forward(self, x):
        mat1 =torch.ones(3, 3)
        return torch.mm(x, mat1)

input = Variable(torch.randn(1, 3))
model = Gemm()
quantize_and_save_model("quantized_gemm", input, model, act_type="int8", wt_type="int8", per_channel=False)

shape = [1, 10, 3]
axis = 1
name = "qlinearsoftmax_v13"
input = torch.rand(shape)
model = nn.Softmax(dim=axis)
# NOTE: this needs latest pytorch to have opset>= 13
quantize_and_save_model(name, input, model, act_type="int8", wt_type="int8", per_channel=False)

def generate_qlinearsoftmax(shape=[1, 10, 3], axis=1, opset=11, name="qlinearsoftmax", previous_name="qlinearsoftmax_v13"):
    from onnx import helper, TensorProto

    # get scales and zero points
    previous_model = onnx.load("models/{}.onnx".format(previous_name))
    previous_model_graph = previous_model.graph
    previous_model_nodes = previous_model_graph.node
    scales = []
    zero_points = []
    def get_val_from_initializers(initializers, name):
        for init in initializers:
            if init.name == name:
                if init.int32_data:
                    return init.int32_data[0]
                elif init.float_data:
                    return init.float_data[0]
                else:
                    raise NotImplementedError()
    scales.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[1]))
    zero_points.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[2]))
    scales.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[3]))
    zero_points.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[4]))


    def make_initializers(input_output_names, scales, zero_points):
        names = [
            input_output_names[0] + "_scale",
            input_output_names[0] + "_zero_point",
            input_output_names[1] + "_scale",
            input_output_names[1] + "_zero_point",
        ]

        initializers = []
        initializers.append(helper.make_tensor(names[0], TensorProto.FLOAT, [], np.array([scales[0]], dtype=np.float32)))
        initializers.append(helper.make_tensor(names[1], TensorProto.INT8, [], np.array([zero_points[0]], dtype=np.int8)))
        initializers.append(helper.make_tensor(names[2], TensorProto.FLOAT, [], np.array([scales[1]], dtype=np.float32)))
        initializers.append(helper.make_tensor(names[3], TensorProto.INT8, [], np.array([zero_points[1]], dtype=np.int8)))

        return initializers

    def make_quantize_dequantize(input_name, output_name, shape, dequantize=False):
        input_names = [
            input_name,
            input_name.split('_')[0] + "_scale",
            input_name.split('_')[0] + "_zero_point",
        ]
        output_names = [
            output_name
        ]

        inputs = []
        if dequantize:
            inputs.append(helper.make_tensor_value_info(input_names[0], TensorProto.INT8, shape))
        else:
            inputs.append(helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape))
        inputs.append(helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info(input_names[2], TensorProto.INT8,  []))

        outputs = []
        if dequantize:
            outputs.append(helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape))
        else:
            outputs.append(helper.make_empty_tensor_value_info(output_names[0]))

        node_type = "DequantizeLinear" if dequantize else "QuantizeLinear"
        node = helper.make_node(node_type, input_names, output_names)

        return node, inputs[0], outputs[0]

    def make_qlinearsoftmax(input_name, output_name, axis, opset):
        input_names = [
            input_name,
            input_name.split('_')[0] + "_scale",
            input_name.split('_')[0] + "_zero_point",
            output_name.split('_')[0] + "_scale",
            output_name.split('_')[0] + "_zero_point",
        ]
        output_names = [
            output_name
        ]

        inputs = []
        inputs.append(helper.make_empty_tensor_value_info(input_names[0]))
        inputs.append(helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info(input_names[2], TensorProto.INT8, []))
        inputs.append(helper.make_tensor_value_info(input_names[3], TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info(input_names[4], TensorProto.INT8, []))

        outputs = []
        outputs.append(helper.make_empty_tensor_value_info(output_names[0]))

        node = helper.make_node("QLinearSoftmax", input_names, output_names, domain="com.microsoft", axis=axis, opset=opset)

        return node, inputs[0], outputs[0]

    input_names = ["input", "input_quantized", "output_quantized"]
    output_names = ["input_quantized", "output_quantized", "output"]

    shared_initializers = make_initializers([input_names[0], output_names[-1]], scales, zero_points)
    node_quantize, graph_input, _ = make_quantize_dequantize(input_names[0], output_names[0], shape)
    node_qsoftmax, _, _ = make_qlinearsoftmax(input_names[1], output_names[1], axis, opset)
    node_dequantize, _, graph_output = make_quantize_dequantize(input_names[2], output_names[2], shape, dequantize=True)

    # create graph
    name = "{}_v{}".format(name, opset)
    graph = helper.make_graph(
        [node_quantize, node_qsoftmax, node_dequantize],
        name,
        [graph_input],
        [graph_output],
        shared_initializers,
    )

    # create model
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra")
    model.opset_import.extend([helper.make_opsetid("com.microsoft", 1)])
    # ignore model check to create non-standard operators
    model_path = "models/{}.onnx".format(name)
    onnx.save(model, model_path)
    print("model is saved to {}".format(model_path))

    sess = rt.InferenceSession(model_path)
    # input = np.random.uniform(-1, 1, sess.get_inputs()[0].shape).astype("float32")
    input = np.load("data/input_{}.npy".format(previous_name)).astype(np.float32)
    output = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name : input})[0]

    print(name + " input has sizes",  input.shape)
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)

    print(name + " output has sizes", output.shape)
    output_files = os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

generate_qlinearsoftmax(shape, axis, name="qlinearsoftmax", previous_name=name)

def generate_quantized_depthwise_conv():
    # 16-group depthwise QLinearConv with non-zero x_zp.
    # Minimal fixture for opencv/opencv#28798: the Conv2Int8 VNNI kernel writes
    # K0=8 output channels per iteration, but groups with Kg<K0 share the same
    # output slot and overwrite each other. ngroups=16, Kg=1 hits that path.
    from onnx import numpy_helper
    from onnx import helper, TensorProto
    name = "quantized_depthwise_conv_int8_weights"
    groups = 16
    in_h = in_w = 8

    np.random.seed(0)
    W   = np.random.randint(-32, 32,   size=(groups, 1, 3, 3), dtype=np.int8)
    B   = np.random.randint(-200, 200, size=(groups,),         dtype=np.int32)
    inp = np.random.uniform(-1.0, 1.0, size=(1, groups, in_h, in_w)).astype(np.float32)

    x_sc = np.float32(0.05); x_zp = np.int8(47)
    w_sc = np.array([0.01], dtype=np.float32); w_zp = np.array([0], dtype=np.int8)
    y_sc = np.float32(0.10); y_zp = np.int8(0)

    inits = [
        numpy_helper.from_array(x_sc, "x_sc"),
        numpy_helper.from_array(x_zp, "x_zp"),
        numpy_helper.from_array(W,    "W"),
        numpy_helper.from_array(w_sc, "w_sc"),
        numpy_helper.from_array(w_zp, "w_zp"),
        numpy_helper.from_array(y_sc, "y_sc"),
        numpy_helper.from_array(y_zp, "y_zp"),
        numpy_helper.from_array(B,    "B"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["X", "x_sc", "x_zp"], ["X_q"]),
        helper.make_node(
            "QLinearConv",
            ["X_q", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp", "B"],
            ["Y_q"],
            group=groups, kernel_shape=[3, 3], pads=[1, 1, 1, 1],
        ),
        helper.make_node("DequantizeLinear", ["Y_q", "y_sc", "y_zp"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(inp.shape))],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(inp.shape))],
        inits,
    )
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                              opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)

    model_path = os.path.join("models", name + ".onnx")
    onnx.save(model, model_path)

    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = sess.run(["Y"], {"X": inp})[0]

    print(name + " input has sizes", inp.shape)
    np.save(os.path.join("data", "input_"  + name), inp)
    print(name + " output has sizes", out.shape)
    np.save(os.path.join("data", "output_" + name), np.ascontiguousarray(out))

generate_quantized_depthwise_conv()

def generate_quantized_bn_mul_add():
    from onnx import numpy_helper
    from onnx import helper, TensorProto
    name = "quantized_bn_mul_add"
    C, H, W = 4, 2, 8
    eps = 1e-5
    mul = np.array([0.5, 0.25, 1.5, -0.5], dtype=np.float32)
    add = np.array([0.5, -0.25, 0.0, 0.75], dtype=np.float32)

    inits = [
        numpy_helper.from_array(np.ones(C, dtype=np.float32),  "gamma"),
        numpy_helper.from_array(np.zeros(C, dtype=np.float32), "beta"),
        numpy_helper.from_array(np.zeros(C, dtype=np.float32), "mean"),
        numpy_helper.from_array(np.ones(C, dtype=np.float32),  "var"),
        numpy_helper.from_array(np.float32(1.0),  "x_sc"),
        numpy_helper.from_array(np.uint8(128),    "x_zp"),
        numpy_helper.from_array((128 + mul / 0.25).astype(np.uint8).reshape(C, 1, 1), "mul_q"),
        numpy_helper.from_array(np.float32(0.25), "mul_sc"),
        numpy_helper.from_array(np.uint8(128),    "mul_zp"),
        numpy_helper.from_array(np.float32(1.0),  "m_sc"),
        numpy_helper.from_array(np.uint8(128),    "m_zp"),
        numpy_helper.from_array((128 + add / 0.25).astype(np.uint8).reshape(C, 1, 1), "add_q"),
        numpy_helper.from_array(np.float32(0.25), "add_sc"),
        numpy_helper.from_array(np.uint8(128),    "add_zp"),
        numpy_helper.from_array(np.float32(1.0),  "y_sc"),
        numpy_helper.from_array(np.uint8(128),    "y_zp"),
    ]
    nodes = [
        helper.make_node("BatchNormalization", ["X", "gamma", "beta", "mean", "var"], ["X_bn"], epsilon=eps),
        helper.make_node("QuantizeLinear", ["X_bn", "x_sc", "x_zp"], ["X_q"]),
        helper.make_node("QLinearMul", ["X_q", "x_sc", "x_zp", "mul_q", "mul_sc", "mul_zp", "m_sc", "m_zp"],
                         ["M_q"], domain="com.microsoft"),
        helper.make_node("QLinearAdd", ["M_q", "m_sc", "m_zp", "add_q", "add_sc", "add_zp", "y_sc", "y_zp"],
                         ["Y_q"], domain="com.microsoft"),
        helper.make_node("DequantizeLinear", ["Y_q", "y_sc", "y_zp"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, C, H, W])],
        inits,
    )
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                              opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)])
    onnx.checker.check_model(model)

    model_path = os.path.join("models", name + ".onnx")
    onnx.save(model, model_path)

    values = np.array([1.4, 2.6, 3.0, 5.0, -1.4, -3.0, 0.6, 7.5,
                       150.0, 200.0, -170.0, -300.0, 127.6, 255.0, 1000.0, -129.0], dtype=np.float32)
    inp = np.broadcast_to(values.reshape(1, 1, H, W), (1, C, H, W)).astype(np.float32).copy()

    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = sess.run(["Y"], {"X": inp})[0]

    bn = inp / np.sqrt(np.float32(1 + eps))
    x_q = np.clip(np.round(bn) + 128, 0, 255) - 128
    m_q = np.clip(np.round(x_q * mul.reshape(1, C, 1, 1)) + 128, 0, 255) - 128
    y_q = np.clip(np.round(m_q + add.reshape(1, C, 1, 1)) + 128, 0, 255) - 128
    folded = np.clip(np.round(bn * mul.reshape(1, C, 1, 1) + add.reshape(1, C, 1, 1)) + 128, 0, 255) - 128
    assert np.array_equal(out, y_q), "ONNX Runtime disagrees with the int8 graph"
    assert not np.array_equal(out, folded), "inputs do not separate the int8 graph from a folded affine"

    print(name + " input has sizes", inp.shape)
    np.save(os.path.join("data", "input_"  + name), inp)
    print(name + " output has sizes", out.shape)
    np.save(os.path.join("data", "output_" + name), np.ascontiguousarray(out))

generate_quantized_bn_mul_add()

def conv_integer_ref(x, w, x_zp, w_zp, pads, strides, dilations, group):
    # int64 reference of ONNX ConvInteger; onnxruntime cross-checks it where it supports the case
    import itertools
    nd = x.ndim - 2
    xs = x.astype(np.int64) - np.int64(x_zp)
    ws = w.astype(np.int64) - np.asarray(w_zp, np.int64).reshape((-1,) + (1,) * (w.ndim - 1))
    xs = np.pad(xs, [(0, 0), (0, 0)] + [(pads[i], pads[i + nd]) for i in range(nd)])
    K, Cg = ws.shape[:2]
    Kg = K // group
    osz = [(xs.shape[2 + i] - dilations[i] * (ws.shape[2 + i] - 1) - 1) // strides[i] + 1
           for i in range(nd)]
    y = np.zeros((x.shape[0], K) + tuple(osz), np.int64)
    for k in range(K):
        g = k // Kg
        for t in itertools.product(*[range(s) for s in ws.shape[2:]]):
            sl = tuple(slice(t[i] * dilations[i], t[i] * dilations[i] + strides[i] * (osz[i] - 1) + 1,
                             strides[i]) for i in range(nd))
            y[:, k] += np.tensordot(xs[(slice(None), slice(g * Cg, (g + 1) * Cg)) + sl],
                                    ws[(k, slice(None)) + t], axes=([1], [0]))
    assert np.abs(y).max() < 2 ** 31
    return y.astype(np.int32)

def random_int(shape, lo, hi, dtype):
    return np.random.randint(lo, hi + 1, size=shape).astype(dtype)

def generate_conv_integer(name, x_shape, x_dtype, w, x_zp=None, w_zp=None, w_input=False,
                          pads=None, strides=None, group=1):
    from onnx import helper, numpy_helper, TensorProto
    onnx_type = {np.uint8: TensorProto.UINT8, np.int8: TensorProto.INT8}
    nd = len(x_shape) - 2
    pads = pads or [0] * (2 * nd)
    strides = strides or [1] * nd
    dilations = [1] * nd

    inputs = [helper.make_tensor_value_info("x", onnx_type[x_dtype], list(x_shape))]
    inits = []
    if w_input:
        inputs.append(helper.make_tensor_value_info("w", onnx_type[w.dtype.type], list(w.shape)))
    else:
        inits.append(numpy_helper.from_array(w, "w"))
    node_inputs = ["x", "w"]
    if x_zp is not None or w_zp is not None:
        node_inputs.append("" if x_zp is None else "x_zero_point")
        if x_zp is not None:
            inits.append(numpy_helper.from_array(np.array(x_zp, dtype=x_dtype), "x_zero_point"))
    if w_zp is not None:
        inits.append(numpy_helper.from_array(np.asarray(w_zp), "w_zero_point"))
        node_inputs.append("w_zero_point")

    node = helper.make_node("ConvInteger", node_inputs, ["y"], pads=pads, strides=strides,
                            dilations=dilations, group=group)
    y_shape = [x_shape[0], w.shape[0]] + [
        (x_shape[2 + i] + pads[i] + pads[i + nd] - dilations[i] * (w.shape[2 + i] - 1) - 1) // strides[i] + 1
        for i in range(nd)]
    graph = helper.make_graph([node], name, inputs,
                              [helper.make_tensor_value_info("y", TensorProto.INT32, y_shape)], inits)
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                              opset_imports=[helper.make_opsetid("", 11)], ir_version=7)
    onnx.checker.check_model(model)
    model_path = os.path.join("models", name + ".onnx")
    onnx.save(model, model_path)

    info = np.iinfo(x_dtype)
    x = random_int(x_shape, info.min, info.max, x_dtype)
    feeds = {"x": x, "w": w} if w_input else {"x": x}
    ref = conv_integer_ref(x, w, 0 if x_zp is None else x_zp, 0 if w_zp is None else w_zp,
                           pads, strides, dilations, group)
    so = rt.SessionOptions()
    so.log_severity_level = 4  # the per-channel w_zero_point refusal below is expected
    try:
        out = rt.InferenceSession(model_path, so, providers=["CPUExecutionProvider"]).run(["y"], feeds)[0]
        assert np.array_equal(out, ref), name
        checked = "checked by onnxruntime"
    except rt.capi.onnxruntime_pybind11_state.Fail:  # onnxruntime only takes a per-tensor w_zero_point
        checked = "numpy reference only"

    for i, vi in enumerate(inputs):
        suffix = "_%d" % i if len(inputs) > 1 else ""
        np.save(os.path.join("data", "input_" + name + suffix), feeds[vi.name])
    np.save(os.path.join("data", "output_" + name), ref)
    print(name, "input", x.shape, "output", ref.shape, checked)

def generate_conv_integer_dynamic(name, x_shape, layers, x_ranges=((-1, 1),), save_data=True):
    # float convs quantized by onnxruntime quantize_dynamic: DynamicQuantizeLinear -> ConvInteger -> Cast -> Mul
    # every x range after the first is saved with suffix _set<i+1> and gives another runtime zero point
    from onnx import helper, numpy_helper, TensorProto
    from onnxruntime.quantization import quantize_dynamic
    nodes, inits, cur = [], [], "x"
    h, w = x_shape[2], x_shape[3]
    for i, (cin, cout, k, stride, pad, group) in enumerate(layers):
        wt = np.random.uniform(-0.5, 0.5, (cout, cin // group, k, k)).astype(np.float32)
        b = np.random.uniform(-0.1, 0.1, cout).astype(np.float32)
        inits += [numpy_helper.from_array(wt, "conv%d_w" % i), numpy_helper.from_array(b, "conv%d_b" % i)]
        nodes.append(helper.make_node("Conv", [cur, "conv%d_w" % i, "conv%d_b" % i], ["conv%d_out" % i],
                                      kernel_shape=[k, k], strides=[stride, stride], pads=[pad] * 4,
                                      group=group))
        cur = "conv%d_out" % i
        h, w = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
        if i + 1 < len(layers):
            nodes.append(helper.make_node("Relu", [cur], ["relu%d_out" % i]))
            cur = "relu%d_out" % i
    out_shape = [x_shape[0], layers[-1][1], h, w]
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
                              [helper.make_tensor_value_info(cur, TensorProto.FLOAT, out_shape)], inits)
    float_model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                                    opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
    onnx.checker.check_model(float_model)
    float_path = os.path.join("models", "dummy.onnx")
    onnx.save(float_model, float_path)
    model_path = os.path.join("models", name + ".onnx")
    quantize_dynamic(float_path, model_path, op_types_to_quantize=["Conv"], weight_type=QuantType.QInt8)
    os.remove(float_path)
    assert any(n.op_type == "ConvInteger" for n in onnx.load(model_path).graph.node), name
    if not save_data:
        print(name, "model only")
        return
    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    for i, (lo, hi) in enumerate(x_ranges):
        x = np.random.uniform(lo, hi, x_shape).astype(np.float32)
        y = sess.run(None, {"x": x})[0]
        suffix = "" if i == 0 else "_set%d" % (i + 1)
        np.save(os.path.join("data", "input_" + name + suffix), x)
        np.save(os.path.join("data", "output_" + name + suffix), np.ascontiguousarray(y))
        print(name + suffix, "input", x.shape, "output", y.shape)

np.random.seed(0)
PAD1 = [1, 1, 1, 1]
generate_conv_integer("convinteger_u8", (1, 20, 11, 9), np.uint8,
                      random_int((12, 20, 3, 3), -127, 127, np.int8), x_zp=131, pads=PAD1)
generate_conv_integer("convinteger_depthwise", (1, 20, 11, 9), np.uint8,
                      random_int((20, 1, 3, 3), 0, 127, np.uint8), x_zp=131, w_zp=np.uint8(60),
                      pads=PAD1, group=20)
generate_conv_integer("convinteger_1x1", (1, 20, 11, 9), np.uint8,
                      random_int((36, 20, 1, 1), -120, 120, np.int8), x_zp=131,
                      w_zp=random_int((36,), -5, 5, np.int8))
generate_conv_integer("convinteger_s8", (1, 20, 11, 9), np.int8,
                      random_int((12, 20, 3, 3), -127, 127, np.int8), x_zp=-9, pads=PAD1)
generate_conv_integer("convinteger_full_range_weights", (1, 20, 11, 9), np.uint8,
                      random_int((12, 20, 3, 3), 0, 255, np.uint8), x_zp=131, w_zp=np.uint8(100),
                      pads=PAD1)
generate_conv_integer("convinteger_groups", (1, 20, 11, 9), np.uint8,
                      random_int((12, 10, 3, 3), -127, 127, np.int8), x_zp=131, pads=PAD1, group=2)
generate_conv_integer("convinteger_runtime_weights", (1, 20, 11, 9), np.uint8,
                      random_int((12, 20, 3, 3), -127, 127, np.int8), x_zp=131, pads=PAD1, w_input=True)
generate_conv_integer("convinteger_1d", (1, 6, 17), np.uint8,
                      random_int((8, 6, 3), -127, 127, np.int8), x_zp=131, pads=[1, 1], strides=[2])
generate_conv_integer_dynamic("convinteger_dynamic", (1, 8, 15, 13),
                              [(8, 16, 3, 1, 1, 1), (16, 16, 3, 1, 1, 16), (16, 24, 1, 1, 0, 1)],
                              x_ranges=((-1, 1), (0, 4)))
generate_conv_integer_dynamic("convinteger_dynamic_block", (1, 3, 256, 256),
                              [(3, 32, 7, 4, 3, 1), (32, 32, 3, 1, 1, 32), (32, 128, 1, 1, 0, 1),
                               (128, 32, 1, 1, 0, 1)], save_data=False)

def matmul_integer_ref(a, b, a_zp, b_zp):
    # int64 reference of ONNX MatMulInteger; an [M] a_zp of a 2-D A has one value per row
    a_zp = np.asarray(a_zp, np.int64)
    if a_zp.ndim == 1 and a.ndim == 2 and a_zp.size > 1:
        a_zp = a_zp.reshape(-1, 1)
    y = np.matmul(a.astype(np.int64) - a_zp, b.astype(np.int64) - np.asarray(b_zp, np.int64))
    assert np.abs(y).max() < 2 ** 31
    return y.astype(np.int32)

def generate_matmul_integer(name, a_shape, a_dtype, b, a_zp=None, b_zp=None, b_input=False,
                            transpose_a=False, a_values=None):
    from onnx import helper, numpy_helper, TensorProto
    onnx_type = {np.uint8: TensorProto.UINT8, np.int8: TensorProto.INT8}
    nd = len(a_shape)
    # transpose_a feeds A through a Transpose of a graph input, so the graph keeps the Transpose
    in_shape = list(a_shape[:-2]) + [a_shape[-1], a_shape[-2]] if transpose_a else list(a_shape)
    inputs = [helper.make_tensor_value_info("a", onnx_type[a_dtype], in_shape)]
    nodes, inits, a_name = [], [], "a"
    if transpose_a:
        nodes.append(helper.make_node("Transpose", ["a"], ["a_t"], perm=list(range(nd - 2)) + [nd - 1, nd - 2]))
        a_name = "a_t"
    if b_input:
        inputs.append(helper.make_tensor_value_info("b", onnx_type[b.dtype.type], list(b.shape)))
    else:
        inits.append(numpy_helper.from_array(b, "b"))
    node_inputs = [a_name, "b"]
    if a_zp is not None or b_zp is not None:
        node_inputs.append("" if a_zp is None else "a_zero_point")
        if a_zp is not None:
            inits.append(numpy_helper.from_array(np.array(a_zp, dtype=a_dtype), "a_zero_point"))
    if b_zp is not None:
        inits.append(numpy_helper.from_array(np.asarray(b_zp, dtype=b.dtype), "b_zero_point"))
        node_inputs.append("b_zero_point")
    nodes.append(helper.make_node("MatMulInteger", node_inputs, ["y"]))
    y_shape = list(np.matmul(np.zeros(a_shape, np.float32), np.zeros(b.shape, np.float32)).shape)
    graph = helper.make_graph(nodes, name, inputs,
                              [helper.make_tensor_value_info("y", TensorProto.INT32, y_shape)], inits)
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                              opset_imports=[helper.make_opsetid("", 11)], ir_version=7)
    onnx.checker.check_model(model)
    model_path = os.path.join("models", name + ".onnx")
    onnx.save(model, model_path)

    info = np.iinfo(a_dtype)
    a_in = random_int(in_shape, info.min, info.max, a_dtype) if a_values is None else a_values
    a = np.swapaxes(a_in, -1, -2) if transpose_a else a_in
    feeds = {"a": a_in, "b": b} if b_input else {"a": a_in}
    ref = matmul_integer_ref(a, b, 0 if a_zp is None else a_zp, 0 if b_zp is None else b_zp)
    so = rt.SessionOptions()
    so.log_severity_level = 4  # the per-row a_zero_point refusal below is expected
    try:
        out = rt.InferenceSession(model_path, so, providers=["CPUExecutionProvider"]).run(["y"], feeds)[0]
        assert np.array_equal(out, ref), name
        checked = "checked by onnxruntime"
    except rt.capi.onnxruntime_pybind11_state.Fail:  # onnxruntime only takes a scalar a_zero_point
        assert np.size(a_zp) > 1, name
        checked = "numpy reference only"

    for i, vi in enumerate(inputs):
        suffix = "_%d" % i if len(inputs) > 1 else ""
        np.save(os.path.join("data", "input_" + name + suffix), feeds[vi.name])
    np.save(os.path.join("data", "output_" + name), ref)
    print(name, "input", a_in.shape, "output", ref.shape, checked)

def generate_matmul_integer_dynamic(name, x_shape, hidden, out, x_ranges=((-1, 1),), save_data=True):
    # float MatMuls quantized by onnxruntime quantize_dynamic: DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul
    # x feeds two MatMuls, so one DynamicQuantizeLinear output feeds two MatMulIntegers
    from onnx import helper, numpy_helper, TensorProto
    from onnxruntime.quantization import quantize_dynamic
    d = x_shape[-1]
    wts = {"w0": (d, hidden), "w1": (d, hidden), "w2": (hidden, out)}
    inits = [numpy_helper.from_array(np.random.uniform(-0.5, 0.5, s).astype(np.float32), n)
             for n, s in wts.items()]
    inits += [numpy_helper.from_array(np.random.uniform(-0.1, 0.1, n).astype(np.float32), "b%d" % i)
              for i, n in enumerate((hidden, out))]
    nodes = [helper.make_node("MatMul", ["x", "w0"], ["m0"]),
             helper.make_node("MatMul", ["x", "w1"], ["m1"]),
             helper.make_node("Add", ["m0", "b0"], ["a0"]),
             helper.make_node("Add", ["a0", "m1"], ["a1"]),
             helper.make_node("Relu", ["a1"], ["r"]),
             helper.make_node("MatMul", ["r", "w2"], ["m2"]),
             helper.make_node("Add", ["m2", "b1"], ["y"])]
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x_shape[:-1]) + [out])],
                              inits)
    float_model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                                    opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
    onnx.checker.check_model(float_model)
    float_path = os.path.join("models", "dummy.onnx")
    onnx.save(float_model, float_path)
    model_path = os.path.join("models", name + ".onnx")
    quantize_dynamic(float_path, model_path, op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
                     per_channel=True)
    os.remove(float_path)
    assert sum(n.op_type == "MatMulInteger" for n in onnx.load(model_path).graph.node) == 3, name
    if not save_data:
        print(name, "model only")
        return
    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    for i, (lo, hi) in enumerate(x_ranges):
        x = np.random.uniform(lo, hi, x_shape).astype(np.float32)
        y = sess.run(None, {"x": x})[0]
        suffix = "" if i == 0 else "_set%d" % (i + 1)
        np.save(os.path.join("data", "input_" + name + suffix), x)
        np.save(os.path.join("data", "output_" + name + suffix), np.ascontiguousarray(y))
        print(name + suffix, "input", x.shape, "output", y.shape)

np.random.seed(1)
generate_matmul_integer("matmulinteger_u8_per_column", (2, 13, 37), np.uint8,
                        random_int((37, 21), 0, 127, np.uint8), a_zp=131,
                        b_zp=random_int((21,), 55, 72, np.uint8))
generate_matmul_integer("matmulinteger_s8_weights", (24, 64), np.uint8,
                        random_int((64, 40), -128, 127, np.int8), a_zp=128, b_zp=np.int8(0))
generate_matmul_integer("matmulinteger_full_range_weights", (20, 48), np.uint8,
                        random_int((48, 24), 0, 255, np.uint8), a_zp=131, b_zp=np.uint8(100))
generate_matmul_integer("matmulinteger_s8", (9, 33), np.int8,
                        random_int((33, 17), -128, 127, np.int8), a_zp=-9,
                        b_zp=random_int((17,), -5, 5, np.int8))
generate_matmul_integer("matmulinteger_runtime_weights", (11, 40), np.uint8,
                        random_int((40, 19), -128, 127, np.int8), a_zp=131, b_input=True)
generate_matmul_integer("matmulinteger_b_zero_point_only", (7, 26), np.uint8,
                        random_int((26, 12), 0, 255, np.uint8), b_zp=np.uint8(90))
generate_matmul_integer("matmulinteger_transposed_a", (13, 40), np.uint8,
                        random_int((40, 10), 0, 255, np.uint8), transpose_a=True)
generate_matmul_integer("matmulinteger_k1_n1", (5, 1), np.uint8,
                        random_int((1, 1), 0, 255, np.uint8))
generate_matmul_integer_dynamic("matmulinteger_dynamic", (1, 50, 64), 96, 40,
                                x_ranges=((-1, 1), (0, 4)))
generate_matmul_integer_dynamic("matmulinteger_dynamic_block", (1, 256, 256), 512, 256, save_data=False)
# rows and columns at the type limits: adjacent (-128)*(-128) products and the largest sums
k = np.arange(16).reshape(16, 1)
limits_u8 = np.concatenate([np.zeros((16, 1)), np.full((16, 1), 255), (k * 17 + np.arange(16)) % 256], 1)
generate_matmul_integer("matmulinteger_extremes", (6, 16), np.uint8, limits_u8.astype(np.uint8),
                        b_zp=np.uint8(255), a_values=limits_u8.T[[0, 1, 2, 0, 1, 3]].astype(np.uint8))
limits_s8 = limits_u8 - 128
generate_matmul_integer("matmulinteger_extremes_s8", (6, 16), np.int8, limits_s8.astype(np.int8),
                        a_zp=127, a_values=limits_s8.T[[0, 1, 2, 0, 1, 3]].astype(np.int8))
# B with batch dims: broadcast against A's, per-batch zero points, a 1-D A
generate_matmul_integer("matmulinteger_batched_b", (2, 3, 5, 19), np.uint8,
                        random_int((3, 19, 13), 0, 255, np.uint8), a_zp=131,
                        b_zp=random_int((3, 1, 13), 100, 140, np.uint8))
generate_matmul_integer("matmulinteger_batched_s8", (1, 4, 9, 24), np.uint8,
                        random_int((1, 4, 24, 10), -111, 127, np.int8), a_zp=0, b_zp=np.int8(0))
generate_matmul_integer("matmulinteger_vector_a", (19,), np.uint8,
                        random_int((3, 19, 13), -128, 127, np.int8), a_zp=131)
# 1-D B, as in numpy.matmul: [M, K] @ [K] -> [M] and [K] @ [K] -> scalar
generate_matmul_integer("matmulinteger_b_vector", (13, 29), np.uint8,
                        random_int((29,), 0, 255, np.uint8), a_zp=131, b_zp=np.uint8(120))
generate_matmul_integer("matmulinteger_vector_vector", (29,), np.uint8,
                        random_int((29,), -128, 127, np.int8), a_zp=131)
# per-row a_zero_point: [M] for a 2-D A, and [D..., M, 1] with a broadcast batch dim for an N-D A
generate_matmul_integer("matmulinteger_a_zp_per_row", (11, 23), np.int8,
                        random_int((23, 9), -128, 127, np.int8),
                        a_zp=random_int((11,), -20, 20, np.int8))
generate_matmul_integer("matmulinteger_a_zp_per_row_nd", (2, 3, 5, 17), np.uint8,
                        random_int((3, 17, 7), 0, 255, np.uint8),
                        a_zp=random_int((2, 1, 5, 1), 100, 160, np.uint8), b_zp=np.uint8(130))

def dynamic_matmul_integer(k, n, a_zp="x_zero_point", weight_scale=True, bias=True):
    # DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul [-> Add] from x to y, a dynamically quantized
    # projection; a_zp is the MatMulInteger a_zero_point, "" for none, and without weight_scale the Mul
    # takes the DynamicQuantizeLinear scale alone
    from onnx import helper, numpy_helper, TensorProto
    inits = [numpy_helper.from_array(random_int((k, n), -128, 127, np.int8), "w")]
    if weight_scale:
        inits.append(numpy_helper.from_array(np.array(0.02, np.float32), "w_scale"))
    inits.append(numpy_helper.from_array(np.array(0, np.int8), "w_zero_point"))
    nodes = [helper.make_node("DynamicQuantizeLinear", ["x"], ["xq", "x_scale", "x_zero_point"]),
             helper.make_node("MatMulInteger", ["xq", "w", a_zp, "w_zero_point"], ["acc"]),
             helper.make_node("Cast", ["acc"], ["acc_f"], to=TensorProto.FLOAT)]
    y0 = "y0" if bias else "y"
    if weight_scale:
        nodes += [helper.make_node("Mul", ["x_scale", "w_scale"], ["scale"]),
                  helper.make_node("Mul", ["acc_f", "scale"], [y0])]
    else:
        nodes.append(helper.make_node("Mul", ["acc_f", "x_scale"], [y0]))
    if bias:
        inits.append(numpy_helper.from_array(np.random.standard_normal(n).astype(np.float32), "bias"))
        nodes.append(helper.make_node("Add", ["bias", "y0"], ["y"]))
    return inits, nodes

def save_with_onnxruntime_reference(name, nodes, inits, x_shape, out, out_dims, save_data=True):
    from onnx import helper, TensorProto
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                                                          ["b", "s", x_shape[-1]])],
                              [helper.make_tensor_value_info(out, TensorProto.FLOAT, out_dims)], inits)
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
    onnx.checker.check_model(model)
    model_path = os.path.join("models", name + ".onnx")
    onnx.save(model, model_path)
    if not save_data:
        print(name, "model only")
        return

    x = np.random.standard_normal(x_shape).astype(np.float32)
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL  # follow the unfused graph
    y = rt.InferenceSession(model_path, so, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    np.save(os.path.join("data", "input_" + name), x)
    np.save(os.path.join("data", "output_" + name), y)
    print(name, "input", x.shape, "output", y.shape)

def generate_matmul_integer_head_split(name, x_shape, n, spec, heads=0, head_dim=0, weight_scale=True,
                                       bias=True, save_data=True):
    # the projection followed by Reshape -> Transpose(0, 2, 1, 3), as in an attention block of a dynamically
    # quantized transformer such as TrOCR; with spec None the Reshape shape [B, S, heads, head_dim] is computed
    # from x's shape at runtime, as exported transformers do
    from onnx import helper, numpy_helper
    inits, nodes = dynamic_matmul_integer(x_shape[-1], n, weight_scale=weight_scale, bias=bias)
    if spec is None:
        inits += [numpy_helper.from_array(np.array([heads, head_dim], np.int64), "head_dims"),
                  numpy_helper.from_array(np.array([0], np.int64), "i0"),
                  numpy_helper.from_array(np.array([1], np.int64), "i1")]
        nodes += [helper.make_node("Shape", ["x"], ["x_shape"]),
                  helper.make_node("Gather", ["x_shape", "i0"], ["batch"], axis=0),
                  helper.make_node("Gather", ["x_shape", "i1"], ["seq"], axis=0),
                  helper.make_node("Concat", ["batch", "seq", "head_dims"], ["spec"], axis=0)]
    else:
        inits.append(numpy_helper.from_array(np.array(spec, np.int64), "spec"))
    nodes += [helper.make_node("Reshape", ["y", "spec"], ["r"]),
              helper.make_node("Transpose", ["r"], ["out"], perm=[0, 2, 1, 3])]
    save_with_onnxruntime_reference(name, nodes, inits, x_shape, "out", ["d0", "d1", "d2", "d3"], save_data)

def generate_matmul_integer_own_zero_point(name, x_shape, n, a_zp):
    # the projection with MatMulInteger given a constant a_zero_point, or none when a_zp is None, instead of
    # the DynamicQuantizeLinear one, which is left unused
    from onnx import numpy_helper
    inits, nodes = dynamic_matmul_integer(x_shape[-1], n, "" if a_zp is None else "a_zero_point")
    if a_zp is not None:
        inits.append(numpy_helper.from_array(np.array(a_zp, np.uint8), "a_zero_point"))
    save_with_onnxruntime_reference(name, nodes, inits, x_shape, "y", ["b", "s", n])

def generate_matmul_integer_static_scale(name, x_shape, n):
    # QuantizeLinear -> MatMulInteger -> Cast -> Mul -> Add with constant scales, as in a statically quantized
    # model: the dequantization scale is one constant per column
    from onnx import helper, numpy_helper, TensorProto
    inits = [numpy_helper.from_array(np.array(0.03, np.float32), "x_scale"),
             numpy_helper.from_array(np.array(128, np.uint8), "x_zero_point"),
             numpy_helper.from_array(random_int((x_shape[-1], n), -128, 127, np.int8), "w"),
             numpy_helper.from_array(np.array(0, np.int8), "w_zero_point"),
             numpy_helper.from_array(np.random.uniform(3e-4, 9e-4, n).astype(np.float32), "scale"),
             numpy_helper.from_array(np.random.standard_normal(n).astype(np.float32), "bias")]
    nodes = [helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["xq"]),
             helper.make_node("MatMulInteger", ["xq", "w", "x_zero_point", "w_zero_point"], ["acc"]),
             helper.make_node("Cast", ["acc"], ["acc_f"], to=TensorProto.FLOAT),
             helper.make_node("Mul", ["acc_f", "scale"], ["y0"]),
             helper.make_node("Add", ["y0", "bias"], ["y"])]
    save_with_onnxruntime_reference(name, nodes, inits, x_shape, "y", ["b", "s", n])

def generate_matmul_integer_per_row_dequant(name, m, k, n):
    # a per-row a_zero_point followed by Cast -> Mul -> Add; onnxruntime rejects a per-row a_zero_point,
    # so the reference is numpy
    from onnx import helper, numpy_helper, TensorProto
    a = random_int((m, k), 0, 255, np.uint8)
    a_zp = random_int((m,), 0, 255, np.uint8)
    b = random_int((k, n), -128, 127, np.int8)
    scale = np.random.uniform(0.001, 0.01, n).astype(np.float32)
    bias = np.random.standard_normal(n).astype(np.float32)
    nodes = [helper.make_node("MatMulInteger", ["a", "b", "a_zero_point"], ["acc"]),
             helper.make_node("Cast", ["acc"], ["acc_f"], to=TensorProto.FLOAT),
             helper.make_node("Mul", ["acc_f", "scale"], ["y0"]),
             helper.make_node("Add", ["y0", "bias"], ["y"])]
    inits = [numpy_helper.from_array(b, "b"), numpy_helper.from_array(a_zp, "a_zero_point"),
             numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(bias, "bias")]
    graph = helper.make_graph(nodes, name, [helper.make_tensor_value_info("a", TensorProto.UINT8, [m, k])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [m, n])], inits)
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra",
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=7)
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join("models", name + ".onnx"))

    y = matmul_integer_ref(a, b, a_zp, 0).astype(np.float32) * scale + bias
    np.save(os.path.join("data", "input_" + name), a)
    np.save(os.path.join("data", "output_" + name), y)
    print(name, "input", a.shape, "output", y.shape, "numpy reference only")

# the MatMulInteger fusion: dequantization and the attention head split fold into one MatMulInt8
np.random.seed(2)
generate_matmul_integer_head_split("matmulinteger_fusion_head_split", (2, 5, 40), 64, [2, 5, 4, 16])
generate_matmul_integer_head_split("matmulinteger_fusion_head_split_runtime_shape", (2, 7, 33), 64, None, 4, 16)
# 0 copies an input dimension and -1 is inferred
generate_matmul_integer_head_split("matmulinteger_fusion_head_split_zero_dims", (3, 9, 17), 48, [0, 0, -1, 16])
# still fused, but computed into a temporary and transposed: head_dim 8 is not a multiple of 16, and
# [1, 3, 8, 16] does not split rows and columns apart
generate_matmul_integer_head_split("matmulinteger_fusion_head_split_d8", (2, 7, 33), 32, None, 4, 8)
generate_matmul_integer_head_split("matmulinteger_fusion_head_split_not_a_split", (1, 6, 20), 64, [1, 3, 8, 16])
# a per-row a_zero_point is applied after the int32 store, so Cast -> Mul -> Add stays unfused
generate_matmul_integer_per_row_dequant("matmulinteger_fusion_per_row_zero_point", 5, 24, 32)
# MatMulInteger takes its own a_zero_point, so the DynamicQuantizeLinear must not fold into it
generate_matmul_integer_own_zero_point("matmulinteger_fusion_dql_const_zero_point", (2, 5, 24), 32, 100)
generate_matmul_integer_own_zero_point("matmulinteger_fusion_dql_no_zero_point", (2, 5, 24), 32, None)
# a constant per-column scale, with no DynamicQuantizeLinear, folds in as well
generate_matmul_integer_static_scale("matmulinteger_fusion_const_scale", (2, 5, 24), 32)
# the Mul takes the DynamicQuantizeLinear scale alone and there is no bias; everything still folds
generate_matmul_integer_head_split("matmulinteger_fusion_head_split_dql_scale_no_bias", (2, 7, 33), 64, None, 4, 16,
                                   weight_scale=False, bias=False)
# perf model only: a cross-attention K/V projection of TrOCR, 768 features into 12 heads of 64
generate_matmul_integer_head_split("matmulinteger_head_split_block", (1, 577, 768), 768, None, 12, 64,
                                   save_data=False)
