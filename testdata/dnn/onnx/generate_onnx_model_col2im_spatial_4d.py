#!/usr/bin/env python3

from pathlib import Path
import onnx
from onnx import TensorProto, checker, helper


def make_col2im_spatial_4d_model(path):
    data_info = helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 16, 1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2, 2, 2])

    image_shape_init = helper.make_tensor("image_shape", TensorProto.INT64, [4], [2, 2, 2, 2])
    block_shape_init = helper.make_tensor("block_shape", TensorProto.INT64, [4], [2, 2, 2, 2])

    node = helper.make_node(
        "Col2Im",
        inputs=["data", "image_shape", "block_shape"],
        outputs=["output"],
        dilations=[1, 1, 1, 1],
        pads=[0, 0, 0, 0, 0, 0, 0, 0],
        strides=[1, 1, 1, 1],
        name="col2im_spatial_4d",
    )

    graph = helper.make_graph(
        [node],
        "col2im_spatial_4d",
        [data_info],
        [output_info],
        initializer=[image_shape_init, block_shape_init],
    )

    model = helper.make_model(
        graph,
        producer_name="col2im-tests",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_col2im_spatial_4d_model(output_dir / "col2im_spatial_4d.onnx")
