"""
Generates SDPA ONNX models + reference I/O .npy for the OpenCV DNN test suite.

The saved sdpa_*.onnx is a single-node graph using a custom "SDPA" op (Q, KT, V
-> Y) that the OpenCV ONNX importer routes directly to SDPALayer. The reference
output is generated separately by an external framework (onnxruntime) running
the mathematically equivalent decomposed subgraph:

    matmul_qk    = MatMul(Q, KT)         # (B, H, S_q, S_kv)
    qk_scaled    = Div(matmul_qk, scale)
    attn         = Softmax(qk_scaled, axis=-1)
    matmul_qkv   = MatMul(attn, V)       # (B, H, S_q, D_v)
    transposed   = Transpose(matmul_qkv, perm=[0,2,1,3])
    Y            = Reshape(transposed, [B, S_q, H*D_v])

Only the single-node SDPA model and the Q/KT/V/Y npy arrays are written to disk.
"""

import os
import math
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def build_sdpa_model(name, B, H, S_q, S_kv, D, D_v, scale):
    """One-node SDPA model. Not runnable in onnxruntime; consumed by OpenCV."""
    Q  = helper.make_tensor_value_info("Q",  TensorProto.FLOAT, [B, H, S_q,  D])
    KT = helper.make_tensor_value_info("KT", TensorProto.FLOAT, [B, H, D,    S_kv])
    V  = helper.make_tensor_value_info("V",  TensorProto.FLOAT, [B, H, S_kv, D_v])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [B, S_q, H * D_v])

    sdpa_node = helper.make_node(
        "SDPA", ["Q", "KT", "V"], ["Y"],
        scale=float(scale))

    graph = helper.make_graph([sdpa_node], name, [Q, KT, V], [Y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="opencv-sdpa-gen")
    model.ir_version = 8
    return model


def build_reference_model(name, B, H, S_q, S_kv, D, D_v, scale):
    """Decomposed equivalent. Used only to produce the reference Y via ORT."""
    Q  = helper.make_tensor_value_info("Q",  TensorProto.FLOAT, [B, H, S_q,  D])
    KT = helper.make_tensor_value_info("KT", TensorProto.FLOAT, [B, H, D,    S_kv])
    V  = helper.make_tensor_value_info("V",  TensorProto.FLOAT, [B, H, S_kv, D_v])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT, [B, S_q, H * D_v])

    scale_init = numpy_helper.from_array(
        np.array(scale, dtype=np.float32), name=f"{name}_scale")
    shape_init = numpy_helper.from_array(
        np.array([B, S_q, H * D_v], dtype=np.int64), name=f"{name}_out_shape")

    nodes = [
        helper.make_node("MatMul",    ["Q", "KT"],                ["qk"]),
        helper.make_node("Div",       ["qk", scale_init.name],    ["qk_scaled"]),
        helper.make_node("Softmax",   ["qk_scaled"],              ["attn"], axis=-1),
        helper.make_node("MatMul",    ["attn", "V"],              ["qkv"]),
        helper.make_node("Transpose", ["qkv"],                    ["qkv_t"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape",   ["qkv_t", shape_init.name], ["Y"]),
    ]

    graph = helper.make_graph(
        nodes, name + "_ref", [Q, KT, V], [Y],
        initializer=[scale_init, shape_init])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="opencv-sdpa-gen-ref")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def save_model_and_data(name, B, H, S_q, S_kv, D, D_v, scale, seed):
    rng = np.random.default_rng(seed)
    q  = rng.uniform(-1.0, 1.0, size=(B, H, S_q,  D   )).astype(np.float32)
    kt = rng.uniform(-1.0, 1.0, size=(B, H, D,    S_kv)).astype(np.float32)
    v  = rng.uniform(-1.0, 1.0, size=(B, H, S_kv, D_v )).astype(np.float32)

    ref_model = build_reference_model(name, B, H, S_q, S_kv, D, D_v, scale)
    sess = ort.InferenceSession(ref_model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    [y] = sess.run(["Y"], {"Q": q, "KT": kt, "V": v})

    sdpa_model = build_sdpa_model(name, B, H, S_q, S_kv, D, D_v, scale)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    onnx.save(sdpa_model, os.path.join(MODELS_DIR, f"{name}.onnx"))
    np.save(os.path.join(DATA_DIR, f"input_{name}_0.npy"), q)
    np.save(os.path.join(DATA_DIR, f"input_{name}_1.npy"), kt)
    np.save(os.path.join(DATA_DIR, f"input_{name}_2.npy"), v)
    np.save(os.path.join(DATA_DIR, f"output_{name}.npy"),  y)
    print(f"[ok] {name}: Q{list(q.shape)} KT{list(kt.shape)} V{list(v.shape)} -> Y{list(y.shape)} (scale={scale:g})")


if __name__ == "__main__":
    save_model_and_data("sdpa_single_head",
                        B=1, H=1, S_q=8,  S_kv=8,  D=8,  D_v=8,
                        scale=math.sqrt(8.0),  seed=12345)
    save_model_and_data("sdpa_multi_head",
                        B=2, H=4, S_q=16, S_kv=16, D=32, D_v=32,
                        scale=math.sqrt(32.0), seed=67890)
    save_model_and_data("sdpa_cross_attention",
                        B=1, H=2, S_q=12, S_kv=20, D=16, D_v=24,
                        scale=math.sqrt(16.0), seed=24680)
