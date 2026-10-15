"""
Create a minimal split-head detector with exactly four classes.

Both outputs are then shaped [1, N, 4], so their shapes no longer say which one
holds the boxes. The decoder falls back to output order, and this pins that rule:
the first output is the boxes, and Model::setOutputNames() can reorder them.

The transposed sibling carries the same values as [1, 4, N], the orientation a
channels-first head splits into. Both must decode identically, which is what
says the two halves are brought to one orientation before either is identified.

The constants are routed through a zero-weighted reduction of the input so the
graph has real layers; a pure initializer graph leaves detect() with no last
layer to inspect. N is 10 so the merged tensor is [1, 10, 8] and stays
non-square, which is what lets the decoder tell the anchor axis from the
channel axis.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

N, NUM_CLASSES, SIDE = 10, 4, 640

# Centre-form boxes in blob pixels, far enough apart that NMS keeps every one.
boxes = np.zeros((1, N, 4), np.float32)
for i in range(N):
    boxes[0, i] = (50.0 + i * 60.0, 50.0 + i * 60.0, 40.0, 40.0)

# One clear winner per row, cycling through the four classes.
scores = np.full((1, N, NUM_CLASSES), 0.1, np.float32)
for i in range(N):
    scores[0, i, i % NUM_CLASSES] = 0.9


def build_model(transposed=False):
    inits = [
        numpy_helper.from_array(boxes, "boxes_const"),
        numpy_helper.from_array(scores, "scores_const"),
        numpy_helper.from_array(np.zeros((1, 1, 1), np.float32), "zero"),
        numpy_helper.from_array(np.array([1, 1, 1], np.int64), "shape3"),
    ]
    # Only the transposed variant needs the extra pair of nodes, so the plain one is unchanged.
    bn, sn = ("boxes_t", "scores_t") if transposed else ("boxes", "scores")
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean4"], axes=[0, 1, 2, 3], keepdims=1),
        helper.make_node("Reshape", ["mean4", "shape3"], ["mean3"]),
        helper.make_node("Mul", ["mean3", "zero"], ["bias"]),
        helper.make_node("Add", ["boxes_const", "bias"], [bn]),
        helper.make_node("Add", ["scores_const", "bias"], [sn]),
    ]
    if transposed:
        nodes += [helper.make_node("Transpose", [bn], ["boxes"], perm=[0, 2, 1]),
                  helper.make_node("Transpose", [sn], ["scores"], perm=[0, 2, 1])]
    shp = (lambda c: [1, c, N]) if transposed else (lambda c: [1, N, c])
    graph = helper.make_graph(
        nodes, "split_head_4class",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIDE, SIDE])],
        outputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, shp(4)),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, shp(NUM_CLASSES)),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build_model(), "models/split_head_4class.onnx")
    onnx.save(build_model(transposed=True), "models/split_head_4class_transposed.onnx")
