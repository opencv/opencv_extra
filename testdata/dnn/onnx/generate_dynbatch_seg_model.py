"""
Create a minimal dynamic-batch instance-segmentation head.

No published YOLO-seg export carries a dynamic batch dimension, so the batched
segmentInstances() path has no real model to test against. This one emits the two outputs
the decoder expects, a [B, N, 4+1+1+nm] detect head and [B, nm, mh, mw] prototype bank.

Both are keyed off the per-sample maximum of the input, so feeding a black frame and a
white frame gives each sample its own boxes and its own prototypes: a decoder that reads
sample 0 for every frame produces visibly wrong masks rather than passing by luck.
Every box crosses a letterbox content edge on purpose, so the test sees the clipping path.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

N, NM, MH, MW, SIDE = 12, 4, 16, 16, 64

# [x1, y1, x2, y2, score, classIdx, mask coefficients], corners in blob pixels.
det = np.zeros((1, N, 6 + NM), np.float32)
det[0, 0] = [10, 10, 30, 30, 0.9, 0, 1, 0, 0, 0]
det[0, 1] = [40, 40, 60, 60, 0.8, 1, 0, 1, 0, 0]  # rows 2.. stay zero-padded

# A white sample slides both boxes four pixels right; a black one leaves them alone.
detDelta = np.zeros_like(det)
detDelta[0, 0, [0, 2]] = 4
detDelta[0, 1, [0, 2]] = 4

# Stripes, so a mask cropped at the wrong offset changes its pixel count. Vertical on
# channel 0 and horizontal on channel 1; a white sample swaps the two.
stripe = np.where(np.arange(MW) % 2 == 0, 8.0, -8.0).astype(np.float32)
vertical = stripe[None, :].repeat(MH, 0)
horizontal = stripe[:, None].repeat(MW, 1)
proto = np.zeros((1, NM, MH, MW), np.float32)
proto[0, 0], proto[0, 1] = vertical, horizontal
protoDelta = np.zeros_like(proto)
protoDelta[0, 0] = horizontal - vertical
protoDelta[0, 1] = vertical - horizontal


def build_model():
    inits = [
        numpy_helper.from_array(det, "det_const"),
        numpy_helper.from_array(detDelta, "det_delta"),
        numpy_helper.from_array(proto, "proto_const"),
        numpy_helper.from_array(protoDelta, "proto_delta"),
        numpy_helper.from_array(np.array([1, 2], np.int64), "ax12"),
        numpy_helper.from_array(np.array([1, 2, 3], np.int64), "ax123"),
    ]
    nodes = [
        helper.make_node("ReduceMax", ["images"], ["perSample"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["perSample", "ax12"], ["s3"]),
        helper.make_node("Unsqueeze", ["perSample", "ax123"], ["s4"]),
        helper.make_node("Mul", ["det_delta", "s3"], ["detShift"]),
        helper.make_node("Mul", ["proto_delta", "s4"], ["protoShift"]),
        helper.make_node("Add", ["det_const", "detShift"], ["det"]),
        helper.make_node("Add", ["proto_const", "protoShift"], ["proto"]),
    ]
    graph = helper.make_graph(
        nodes, "dynbatch_seg",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                              ["batch", 3, SIDE, SIDE])],
        outputs=[
            helper.make_tensor_value_info("det", TensorProto.FLOAT, ["batch", N, 6 + NM]),
            helper.make_tensor_value_info("proto", TensorProto.FLOAT, ["batch", NM, MH, MW]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build_model(), "models/dynbatch_seg.onnx")
