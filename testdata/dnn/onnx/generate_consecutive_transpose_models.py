#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper


INPUT_SHAPE = [1, 3, 4, 4]
OPSET_VERSION = 13
IR_VERSION = 7
PRODUCER_NAME = "opencv_dnn_consecutive_transpose_test"


def apply_perm(shape, perm):
    if perm is None:
        return list(reversed(shape))
    return [shape[i] for i in perm]


def make_transpose_node(name, input_name, output_name, perm):
    if perm is None:
        return helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            name=name,
        )

    return helper.make_node(
        "Transpose",
        inputs=[input_name],
        outputs=[output_name],
        name=name,
        perm=perm,
    )


def make_model(filename, first_perm, second_perm, output_dir):
    first_shape = apply_perm(INPUT_SHAPE, first_perm)
    output_shape = apply_perm(first_shape, second_perm)

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, INPUT_SHAPE
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )
    first_output_info = helper.make_tensor_value_info(
        "transpose_1_output", TensorProto.FLOAT, first_shape
    )

    nodes = [
        make_transpose_node(
            "transpose_1", "input", "transpose_1_output", first_perm
        ),
        make_transpose_node(
            "transpose_2", "transpose_1_output", "output", second_perm
        ),
    ]

    graph = helper.make_graph(
        nodes,
        name=Path(filename).stem,
        inputs=[input_info],
        outputs=[output_info],
        value_info=[first_output_info],
    )

    model = helper.make_model(
        graph,
        producer_name=PRODUCER_NAME,
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )
    model.ir_version = IR_VERSION
    checker.check_model(model)

    path = output_dir / filename
    onnx.save(model, path)
    print("Wrote %s (%d bytes)" % (path, os.path.getsize(path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent / "models",
        type=Path,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    make_model(
        "transpose_identity.onnx",
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        args.output_dir,
    )
    make_model(
        "transpose_default_perm.onnx",
        None,
        None,
        args.output_dir,
    )
    make_model(
        "transpose_non_identity.onnx",
        [0, 2, 3, 1],
        [0, 1, 3, 2],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
