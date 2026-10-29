#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Regenerate quantization scale validation models in models/ without dependencies.

Run ``python3 testdata/dnn/onnx/generate_onnx_quantize_scalar_models.py`` from
the opencv_extra root. ``--models DIR`` selects another output directory.
The encoder writes only the protobuf fields these small models need; field
numbers follow ONNX's ModelProto, GraphProto, NodeProto, TensorProto, and
ValueInfoProto definitions.
"""

import argparse
import hashlib
import struct
from pathlib import Path

FLOAT = 1
INT8 = 3


def varint(value):
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while value > 127:
        out.append((value & 0x7f) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def vint(number, value):
    return varint(number << 3) + varint(value)


def data(number, value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return varint((number << 3) | 2) + varint(len(value)) + value


def tensor(name, dims, element_type, payload):
    # TensorProto: dims=1, data_type=2, name=8, raw_data=9.
    return (b''.join(vint(1, dimension) for dimension in dims)
            + vint(2, element_type) + data(8, name) + data(9, payload))


def value_info(name, element_type, dims):
    # ValueInfoProto -> TypeProto -> TypeProto.Tensor -> TensorShapeProto.
    shape = b''.join(data(1, vint(1, dimension)) for dimension in dims)
    tensor_type = vint(1, element_type) + data(2, shape)
    return data(1, name) + data(2, data(1, tensor_type))


def quantize_linear(scale_type, scale_payload, scale_dims=None):
    scale_dims = [] if scale_dims is None else scale_dims
    tensor_dims = [1, 4] if not scale_dims else [1, scale_dims[0]]
    zero_points = b'\x00' * (scale_dims[0] if scale_dims else 1)
    initializers = [tensor('y_scale', scale_dims, scale_type, scale_payload),
                    tensor('y_zero_point', scale_dims, INT8, zero_points)]
    # NodeProto: input=1, output=2, name=3, op_type=4.
    node = (data(1, 'x') + data(1, 'y_scale') + data(1, 'y_zero_point')
            + data(2, 'y') + data(3, 'quantize') + data(4, 'QuantizeLinear'))
    # GraphProto: node=1, name=2, initializer=5, input=11, output=12.
    graph = (data(1, node) + data(2, 'quantize_scalar')
             + b''.join(data(5, item) for item in initializers)
             + data(11, value_info('x', FLOAT, tensor_dims))
             + data(12, value_info('y', INT8, tensor_dims)))
    # ModelProto: ir_version=1, graph=7, opset_import=8.
    return vint(1, 8) + data(7, graph) + data(8, vint(2, 13))


def qlinear_conv(weight_scale_type, weight_scale_payload):
    initializers = [
        tensor('x_scale', [], FLOAT, struct.pack('<f', 0.5)),
        tensor('x_zero_point', [], INT8, b'\x00'),
        tensor('weights', [2, 1, 1, 1], INT8, b'\x01\x02'),
        tensor('w_scale', [2], weight_scale_type, weight_scale_payload),
        tensor('w_zero_point', [2], INT8, b'\x00\x00'),
        tensor('y_scale', [], FLOAT, struct.pack('<f', 0.25)),
        tensor('y_zero_point', [], INT8, b'\x00'),
    ]
    inputs = ['x', 'x_scale', 'x_zero_point', 'weights', 'w_scale',
              'w_zero_point', 'y_scale', 'y_zero_point']
    node = (b''.join(data(1, item) for item in inputs)
            + data(2, 'y') + data(3, 'qlinearconv') + data(4, 'QLinearConv'))
    graph = (data(1, node) + data(2, 'qlinearconv_weight_scale')
             + b''.join(data(5, item) for item in initializers)
             + data(11, value_info('x', INT8, [1, 1, 1, 1]))
             + data(12, value_info('y', INT8, [1, 2, 1, 1])))
    return vint(1, 8) + data(7, graph) + data(8, vint(2, 13))


def cases():
    return [
        ('quantize_scalar_control-float-scale.onnx',
         'QuantizeLinear, FLOAT scalar y_scale 0.5, INT8 scalar zero point, accepted',
         quantize_linear(FLOAT, struct.pack('<f', 0.5))),
        ('quantize_scalar_invalid-int8-scale.onnx',
         'QuantizeLinear, INT8 scalar y_scale (1 byte read as a 4-byte float), rejected',
         quantize_linear(INT8, b'\x02')),
        ('quantize_per_axis_invalid-int8-scale.onnx',
         'QuantizeLinear, INT8 two-element y_scale read as float values, rejected',
         quantize_linear(INT8, b'\x02\x03', [2])),
        ('qlinearconv_invalid-int8-weight-scale.onnx',
         'QLinearConv, INT8 two-element weight scale read as float values, rejected',
         qlinear_conv(INT8, b'\x02\x03')),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=Path(__file__).resolve().with_name('models'))
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    for filename, _, blob in cases():
        (args.models / filename).write_bytes(blob)
        print('%s  %s' % (hashlib.sha256(blob).hexdigest(), filename))


if __name__ == '__main__':
    main()
