import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import os.path
import onnx
import google.protobuf.text_format
import io


def assertExpected(s):
    if not (isinstance(s, str) or (sys.version_info[0] == 2 and isinstance(s, unicode))):
        raise TypeError("assertExpected is strings only")

def assertONNXExpected(binary_pb):
    model_def = onnx.ModelProto.FromString(binary_pb)
    onnx.checker.check_model(model_def)
    # doc_string contains stack trace in it, strip it
    onnx.helper.strip_doc_string(model_def)
    assertExpected(google.protobuf.text_format.MessageToString(model_def, float_format='.15g'))
    return model_def


def export_to_string(model, inputs):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f)
    return f.getvalue()


def save_data_and_model(name, input, model):
    print name + " input has sizes",  input.shape
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)

    output = model(input)
    print name + " output has sizes", output.shape
    print
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, output.data)

    models_files = os.path.join("models", name + ".onnx")

    onnx_model_pb = export_to_string(model, input)
    model_def = assertONNXExpected(onnx_model_pb)
    with open(models_files, 'wb') as file:
        file.write(model_def.SerializeToString())

torch.manual_seed(0)

input = Variable(torch.randn(1, 3, 10, 20))
max_pool = nn.MaxPool2d(kernel_size=(5,3), stride=1, padding=0, dilation=1)
save_data_and_model("maxpooling", input, max_pool)


input = Variable(torch.randn(1, 3, 10, 20))
conv = nn.Conv2d(3, 5, kernel_size=5, stride=2, padding=0)
save_data_and_model("convolution", input, conv)


input = Variable(torch.randn(2, 3))
linear = nn.Linear(3, 4, bias=True)
save_data_and_model("linear", input, linear)

input = Variable(torch.randn(2, 3, 12, 18))
maxpooling_sigmoid = nn.Sequential(
          nn.MaxPool2d(kernel_size=4, stride=2, padding=(1, 2), dilation=1),
          nn.Sigmoid()
        )
save_data_and_model("maxpooling_sigmoid", input, maxpooling_sigmoid)


input = Variable(torch.randn(1, 3, 10, 20))
conv2 = nn.Sequential(
          nn.Conv2d(3, 6, kernel_size=(5,3), stride=1, padding=1),
          nn.Conv2d(6, 4, kernel_size=5, stride=2, padding=(0,2))
          )
save_data_and_model("convolution2", input, conv2)


input = Variable(torch.randn(2, 3, 30, 45))
maxpool2 = nn.Sequential(
           nn.MaxPool2d(kernel_size=5, stride=1, padding=0, dilation=1),
           nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1)
           )
save_data_and_model("maxpooling2", input, maxpool2)


input = Variable(torch.randn(1, 2, 10, 10))
relu = nn.ReLU(inplace=True)
save_data_and_model("ReLU", input, relu)


input = Variable(torch.randn(2, 3))
dropout = nn.Dropout()
dropout.eval()
save_data_and_model("dropout", input, dropout)
