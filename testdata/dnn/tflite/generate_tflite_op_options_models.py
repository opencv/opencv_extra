#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Generate TFLite importer models for operator option validation with flatc.

Pass OpenCV's modules/dnn/src/tflite/schema.fbs with --schema. Its namespace
is opencv_tflite. Run from the opencv_extra root, for example:

    python3 testdata/dnn/tflite/generate_tflite_op_options_models.py \\
        --schema ../opencv/modules/dnn/src/tflite/schema.fbs --out testdata/dnn/tflite

Custom operator options are the raw C structs OpenCV reads
(TfLitePoolParams up to and including `activation`, as MediaPipe writes them,
and TfLiteTransposeConvParams), little-endian int32 fields.
"""

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path


SCHEMA_SHA256 = "caae52a01701dce1949e2f591176583901e9ebd161336259838f0f83ddaf74ea"

PADDING_SAME = 1
MAX_POOL_2D = 17
CUSTOM = 32


def tensor(name, shape, buffer=0):
    return {"shape": shape, "type": "FLOAT32", "buffer": buffer, "name": name, "has_rank": True}


def pool_params(padding=PADDING_SAME, stride_w=2, stride_h=2, filter_w=2, filter_h=2, activation=0):
    # TfLitePoolParams fields up to and including `activation` (24 bytes).
    return list(struct.pack("<6i", padding, stride_w, stride_h, filter_w, filter_h, activation))


def transpose_conv_params(padding=PADDING_SAME, stride_w=2, stride_h=2):
    # TfLiteTransposeConvParams (12 bytes).
    return list(struct.pack("<3i", padding, stride_w, stride_h))


def model(name, operator_code, tensors, inputs, outputs, op, buffers=None):
    op = dict(op)
    op.setdefault("opcode_index", 0)
    return {
        "version": 3,
        "operator_codes": [operator_code],
        "subgraphs": [{"tensors": tensors, "inputs": inputs, "outputs": outputs,
                       "operators": [op], "name": name}],
        "description": name,
        "buffers": buffers if buffers is not None else [{}],
    }


def custom_code(code):
    return {"deprecated_builtin_code": CUSTOM, "custom_code": code, "version": 1, "builtin_code": "CUSTOM"}


def argmax_pool(options):
    return model("argmax_pool", custom_code("MaxPoolingWithArgmax2D"),
                 [tensor("input", [1, 4, 4, 1]), tensor("output", [1, 2, 2, 1]),
                  tensor("indices", [1, 2, 2, 1])],
                 [0], [1, 2],
                 {"inputs": [0], "outputs": [1, 2], "custom_options": options})


def unpooling(options):
    return model("unpooling", custom_code("MaxUnpooling2D"),
                 [tensor("input", [1, 2, 2, 1]), tensor("indices", [1, 2, 2, 1]),
                  tensor("output", [1, 4, 4, 1])],
                 [0, 1], [2],
                 {"inputs": [0, 1], "outputs": [2], "custom_options": options})


def transpose_conv(options):
    weights = list(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    bias = list(struct.pack("<f", 0.5))
    return model("transpose_conv", custom_code("Convolution2DTransposeBias"),
                 [tensor("input", [1, 2, 2, 1]), tensor("weights", [1, 2, 2, 1], 1),
                  tensor("bias", [1], 2), tensor("output", [1, 4, 4, 1])],
                 [0], [3],
                 {"inputs": [0, 1, 2], "outputs": [3], "custom_options": options},
                 buffers=[{}, {"data": weights}, {"data": bias}])


def max_pool(stride_w=2, opcode_index=0):
    return model("max_pool",
                 {"deprecated_builtin_code": MAX_POOL_2D, "version": 1, "builtin_code": "MAX_POOL_2D"},
                 [tensor("input", [1, 4, 4, 1]), tensor("output", [1, 2, 2, 1])],
                 [0], [1],
                 {"opcode_index": opcode_index, "inputs": [0], "outputs": [1],
                  "builtin_options_type": "Pool2DOptions",
                  "builtin_options": {"padding": "SAME", "stride_w": stride_w, "stride_h": 2,
                                      "filter_width": 2, "filter_height": 2,
                                      "fused_activation_function": "NONE"}})


def cases():
    return [
        ("tflite_argmax_pool_control", "MaxPoolingWithArgmax2D, 24-byte options, stride 2, accepted",
         argmax_pool(pool_params())),
        ("tflite_argmax_pool_short_options", "MaxPoolingWithArgmax2D, 4-byte options, rejected",
         argmax_pool(pool_params()[:4])),
        ("tflite_argmax_pool_zero_stride", "MaxPoolingWithArgmax2D, 24-byte options, stride_w 0, rejected",
         argmax_pool(pool_params(stride_w=0))),
        ("tflite_unpooling_control", "MaxUnpooling2D, 24-byte options, stride 2, accepted",
         unpooling(pool_params())),
        ("tflite_unpooling_short_options", "MaxUnpooling2D, 4-byte options, rejected",
         unpooling(pool_params()[:4])),
        ("tflite_transpose_conv_control", "Convolution2DTransposeBias, 12-byte options, accepted",
         transpose_conv(transpose_conv_params())),
        ("tflite_transpose_conv_short_options", "Convolution2DTransposeBias, 4-byte options, rejected",
         transpose_conv(transpose_conv_params()[:4])),
        ("tflite_transpose_conv_zero_stride", "Convolution2DTransposeBias, stride_w 0, rejected",
         transpose_conv(transpose_conv_params(stride_w=0))),
        ("tflite_max_pool_control", "MAX_POOL_2D, stride 2, accepted",
         max_pool()),
        ("tflite_max_pool_zero_stride", "MAX_POOL_2D, stride_w 0, rejected",
         max_pool(stride_w=0)),
        ("tflite_opcode_index_out_of_range", "MAX_POOL_2D operator with opcode_index 3 of 1 operator code, rejected",
         max_pool(opcode_index=3)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flatc", default="flatc")
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if hashlib.sha256(args.schema.read_bytes()).hexdigest() != SCHEMA_SHA256:
        parser.error("schema hash differs from the pinned OpenCV TFLite schema")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tflite-op-options-") as temporary:
        for name, _, m in cases():
            json_path = Path(temporary) / (name + ".json")
            json_path.write_text(json.dumps(m, separators=(",", ":")) + "\n", encoding="utf-8")
            subprocess.run([args.flatc, "-b", "--strict-json", "-o", str(args.out),
                            str(args.schema), str(json_path)], check=True)
            blob = (args.out / (name + ".tflite")).read_bytes()
            print("%s  %s.tflite" % (hashlib.sha256(blob).hexdigest(), name))


if __name__ == "__main__":
    main()
