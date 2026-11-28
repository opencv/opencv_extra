#!/usr/bin/env python3

from pathlib import Path
from onnx import TensorProto, checker, helper, save


def make_tensorscatter_inplace_model(path):
    past = helper.make_tensor_value_info("past", TensorProto.FLOAT, [1, 5, 2])
    update = helper.make_tensor_value_info("update", TensorProto.FLOAT, [1, 2, 2])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 5, 2])

    relu_in = helper.make_node("Relu", inputs=["past"], outputs=["past_tmp"], name="relu_in")
    scatter = helper.make_node(
        "TensorScatter",
        inputs=["past_tmp", "update", "indices"],
        outputs=["scattered"],
        axis=1,
        mode="circular",
        name="ts",
    )
    relu_out = helper.make_node("Relu", inputs=["scattered"], outputs=["out"], name="relu_out")

    graph = helper.make_graph(
        [relu_in, scatter, relu_out],
        "ts_inplace_graph",
        [past, update, indices],
        [out],
    )

    model = helper.make_model(
        graph,
        producer_name="opencv-test",
        opset_imports=[helper.make_operatorsetid("", 24)],
    )
    checker.check_model(model)
    save(model, path)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_tensorscatter_inplace_model(output_dir / "tensorscatter_inplace.onnx")

