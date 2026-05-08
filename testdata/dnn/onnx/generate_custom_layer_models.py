#!/usr/bin/env python3
# Generates two tiny ONNX models with non-standard ops + reference IO,
# used by the new-engine custom-layer test.

import os
import numpy as np
import onnx
from onnx import helper, TensorProto

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(OUT_DIR, "models")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def make_custom_op_model(name, op_type, domain, scale, bias, shape=(1, 3, 4, 4)):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))

    node = helper.make_node(
        op_type, inputs=["input"], outputs=["output"], domain=domain,
        scale=float(scale), bias=float(bias), name=name,
    )

    graph = helper.make_graph([node], name, [inp], [out])
    opset_imports = [helper.make_operatorsetid("", 13)]
    if domain:
        opset_imports.append(helper.make_operatorsetid(domain, 1))

    model = helper.make_model(graph, opset_imports=opset_imports,
                              producer_name="opencv_test_custom_layer")
    model.ir_version = 7

    # Custom ops aren't in the registered schema; skip checker.
    model_path = os.path.join(MODEL_DIR, name + ".onnx")
    onnx.save(model, model_path)
    print(f"Wrote {model_path}")

    rng = np.random.RandomState(0)
    x = rng.randn(*shape).astype(np.float32)
    y = scale * x + bias
    np.save(os.path.join(DATA_DIR, "input_" + name + ".npy"), x)
    np.save(os.path.join(DATA_DIR, "output_" + name + ".npy"), y)


make_custom_op_model("custom_layer_default_domain",
                     op_type="MyCustomOp", domain="", scale=2.5, bias=0.5)
make_custom_op_model("custom_layer_custom_domain",
                     op_type="MyDomainOp", domain="my.namespace",
                     scale=-1.5, bias=0.25)
