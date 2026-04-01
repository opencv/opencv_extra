"""
Create a minimal ONNX model that reproduces the empty-tensor reshape
crash fixed in layers_common.cpp. NMS with high score threshold produces
K=0 detections, then Reshape [0,3] -> [1,0,3] triggers the crash.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def make_const(name, dtype, values):
    data = np.array(values, dtype=np.float32 if dtype == TensorProto.FLOAT else np.int64)
    return numpy_helper.from_array(data, name)


def build_model():
    inits = [
        make_const("max_out", TensorProto.INT64, 10),
        make_const("iou_thr", TensorProto.FLOAT, 0.5),
        make_const("score_thr", TensorProto.FLOAT, 1e10),
        make_const("shape", TensorProto.INT64, [1, -1, 3]),
    ]

    nodes = [
        helper.make_node("NonMaxSuppression",
                         ["boxes", "scores", "max_out", "iou_thr", "score_thr"],
                         ["nms_out"]),
        helper.make_node("Reshape", ["nms_out", "shape"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, "nms_reshape_empty",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 10, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 10]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.INT64, [1, None, 3]),
        ],
        initializer=inits,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)

    path = "nms_reshape_empty.onnx"
    onnx.save(model, path)

    import os
    print(f"Saved: {path} ({os.path.getsize(path)} bytes)")


def generate_inputs():
    rng = np.random.RandomState(42)
    np.save("input_nms_reshape_empty_0.npy", rng.randn(1, 10, 4).astype(np.float32) * 10)
    np.save("input_nms_reshape_empty_1.npy", np.full((1, 1, 10), 0.01, dtype=np.float32))
    print("Saved 2 input .npy files")


if __name__ == "__main__":
    build_model()
    generate_inputs()
