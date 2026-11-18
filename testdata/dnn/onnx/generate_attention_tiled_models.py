import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper

# Models for Layer_AttentionTiledCUDA: a decomposed BERT self-attention that the 5.x graph engine fuses into one
# Attention layer, with an optional additive mask input "M". MultiTileAccuracy compares the CUDA output with the
# CPU one at S = 520 (three query tiles); SubquadraticWorkspace measures device memory at S = 2048.


def save_attention(path, B, S, H, Dh, w, bias, mask_dims=None):
    C = H * Dh
    nodes = []
    for name, cols, perm in (("Q", slice(0, C), [0, 2, 1, 3]),
                             ("K", slice(C, 2 * C), [0, 2, 3, 1]),
                             ("V", slice(2 * C, 3 * C), [0, 2, 1, 3])):
        nodes += [
            helper.make_node("MatMul", ["X", "W" + name.lower()], [name + "m"]),
            helper.make_node("Add", [name + "m", "B" + name.lower()], [name + "a"]),
            helper.make_node("Reshape", [name + "a", "qkv_shape"], [name + "r"]),
            helper.make_node("Transpose", [name + "r"], [name + "t"], perm=perm),
        ]
    nodes += [
        helper.make_node("MatMul", ["Qt", "Kt"], ["Scores"]),
        helper.make_node("Div", ["Scores", "Scale"], ["Scaled"]),
    ]
    softmax_input = "Scaled"
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, S, C])]
    if mask_dims:
        nodes.append(helper.make_node("Add", ["Scaled", "M"], ["Masked"]))
        softmax_input = "Masked"
        inputs.append(helper.make_tensor_value_info("M", TensorProto.FLOAT, mask_dims))
    nodes += [
        helper.make_node("Softmax", [softmax_input], ["Probs"], axis=-1),
        helper.make_node("MatMul", ["Probs", "Vt"], ["Ctx"]),
        helper.make_node("Transpose", ["Ctx"], ["CtxT"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["CtxT", "out_shape"], ["Y"]),
    ]
    initializers = []
    for i, name in enumerate("qkv"):
        initializers.append(numpy_helper.from_array(w[:, i * C:(i + 1) * C].copy(), "W" + name))
        initializers.append(numpy_helper.from_array(bias[i * C:(i + 1) * C].copy(), "B" + name))
    initializers += [
        numpy_helper.from_array(np.array([np.sqrt(Dh)], dtype=np.float32), "Scale"),
        numpy_helper.from_array(np.array([B, S, H, Dh], dtype=np.int64), "qkv_shape"),
        numpy_helper.from_array(np.array([B, S, C], dtype=np.int64), "out_shape"),
    ]
    graph = helper.make_graph(nodes, "attention_tiled", inputs,
                              [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, S, C])],
                              initializer=initializers)
    model = helper.make_model(graph, producer_name="attention-tiled-regression",
                              opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 8
    checker.check_model(model)
    onnx.save(model, path)


rng = np.random.default_rng(0)

# S = 520 gives query tiles of 256, 256 and 8 rows; the same weights for every mask layout
B, S, H, Dh = 2, 520, 4, 16
C = H * Dh
w = rng.uniform(-0.2, 0.2, (C, 3 * C)).astype(np.float32)
bias = rng.uniform(-0.1, 0.1, (3 * C,)).astype(np.float32)
for suffix, mask_dims in (("", None),
                          ("_mask_1x1x1xS", [1, 1, 1, S]),
                          ("_mask_1x1xSxS", [1, 1, S, S]),
                          ("_mask_Bx1xSxS", [B, 1, S, S]),
                          ("_mask_BxHxSxS", [B, H, S, S])):
    save_attention("models/attention_tiled" + suffix + ".onnx", B, S, H, Dh, w, bias, mask_dims)

# S = 2048 with 12 heads: dense scores [B*H, S, S] take 384 MiB in FP32; the hidden size only scales the projections
B, S, H, Dh = 2, 2048, 12, 16
C = H * Dh
save_attention("models/attention_tiled_s2048.onnx", B, S, H, Dh,
               rng.uniform(-0.05, 0.05, (C, 3 * C)).astype(np.float32),
               rng.uniform(-0.05, 0.05, (3 * C,)).astype(np.float32))
