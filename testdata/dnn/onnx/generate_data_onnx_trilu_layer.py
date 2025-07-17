from onnx import TensorProto, version_converter
import onnx
from onnx.helper import (
    make_model, make_node, make_graph, make_tensor_value_info,
)
import numpy as np
import onnxruntime as ort
import os

testcases = [
  {
    "lab":"tril",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 1, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[4, 0, 0, 0, 0],
      [1, 2, 0, 0, 0],
      [9, 4, 1, 0, 0],
      [4, 3, 4, 2, 0]],
    "upper": 0,
    "k": 0
  },
  {
    "lab":"tril_neg",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 1, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[0, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [9, 4, 0, 0, 0],
      [4, 3, 4, 0, 0]],
    "upper": 0,
    "k": -1
  },
  {
    "lab":"tril_one_row",
    "X":\
      [[[6, 2, 4, 1, 6]],
      [[8, 3, 8, 7, 0]],
      [[2, 2, 9, 5, 9]]],
    "Y":\
      [[[6, 0, 0, 0, 0]],
      [[8, 0, 0, 0, 0]],
      [[2, 0, 0, 0, 0]]],
    "upper": 0,
  },
  {
    "lab":"tril_one_row1D",
    "X":\
      [6, 2, 4, 1, 6],
    "Y":\
      [6, 0, 0, 0, 0],
    "upper": 0,
  },
  {
    "lab":"tril_out_neg",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 1, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]],
    "upper": 0,
    "k": -7
  },
  {
    "lab":"tril_out_pos",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 1, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 1, 8, 7],
      [4, 3, 4, 2, 4]],
    "upper": 0,
    "k": 6
  },
  {
    "lab":"tril_pos",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 1, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[4, 7, 3, 0, 0],
      [1, 2, 8, 6, 0],
      [9, 4, 1, 8, 7],
      [4, 3, 4, 2, 4]],
    "upper": 0,
    "k": 2
  },
  {
    "lab":"tril_square",
    "X":\
      [[[0, 4, 3],
        [2, 0, 9],
        [8, 2, 5]],
      [[2, 7, 2],
        [2, 6, 0],
        [2, 6, 5]]],
    "Y":\
      [[[0, 0, 0],
        [2, 0, 0],
        [8, 2, 5]],
      [[2, 0, 0],
        [2, 6, 0],
        [2, 6, 5]]],
    "upper": 0,
  },
  {
    "lab":"tril_square_neg",
    "X":\
      [[[0, 4, 3],
        [2, 0, 9],
        [8, 2, 5]],
      [[2, 7, 2],
        [2, 6, 0],
        [2, 6, 5]]],
    "Y":\
      [[[0, 0, 0],
        [2, 0, 0],
        [8, 2, 0]],
      [[0, 0, 0],
        [2, 0, 0],
        [2, 6, 0]]],
    "upper": 0,
    "k": -1
  },
  {
    "lab":"triu",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 0, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[4, 7, 3, 7, 9],
      [0, 2, 8, 6, 9],
      [0, 0, 0, 8, 7],
      [0, 0, 0, 2, 4]],
  },
  {
    "lab":"triu_neg",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 0, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [0, 4, 0, 8, 7],
      [0, 0, 4, 2, 4]],
    "k": -1
  },
  {
    "lab":"triu_one_row",
    "X":\
      [[[1, 4, 9, 7, 1]],
      [[9, 2, 8, 8, 4]],
      [[3, 9, 7, 4, 2]]],
    "Y":\
      [[[0, 4, 9, 7, 1]],
      [[0, 2, 8, 8, 4]],
      [[0, 9, 7, 4, 2]]],
    "k": 1,
  },
  {
    "lab":"triu_out_neg_out",
    "X":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 0, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":\
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 0, 8, 7],
      [4, 3, 4, 2, 4]],
    "k":-7
  },
  {
    "lab":"triu_out_pos",
    "X":
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 0, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":
      [[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]],
    "k": 6
  },
  {
    "lab":"triu_pos",
    "X":
      [[4, 7, 3, 7, 9],
      [1, 2, 8, 6, 9],
      [9, 4, 0, 8, 7],
      [4, 3, 4, 2, 4]],
    "Y":
      [[0, 0, 3, 7, 9],
      [0, 0, 0, 6, 9],
      [0, 0, 0, 0, 7],
      [0, 0, 0, 0, 0]],
    "k":2
  },
  {
    "lab": "triu_square",
    "X":
      [[[4, 6, 9],
        [7, 5, 4],
        [8, 1, 2]],
      [[1, 4, 9],
        [9, 6, 3],
        [8, 9, 8]]],
    "Y":
      [[[4, 6, 9],
        [0, 5, 4],
        [0, 0, 2]],
      [[1, 4, 9],
        [0, 6, 3],
        [0, 0, 8]]]
  },
  {
    "lab": "triu_square_neg",
    "X":
      [[[4, 6, 9],
        [7, 5, 4],
        [8, 1, 2]],
      [[1, 4, 9],
        [9, 6, 3],
        [8, 9, 8]]],
    "Y":
      [[[4, 6, 9],
        [7, 5, 4],
        [0, 1, 2]],
      [[1, 4, 9],
        [9, 6, 3],
        [0, 9, 8]]],
    "k": -1
  }
]

for i in range(len(testcases)):
    data_case = testcases[i]
    k = data_case.get("k", 0)
    upper = data_case.get("upper", 1)
    lab= data_case.get("lab", "tril")

    input= np.array(data_case["X"], dtype=np.float32)
    output= np.array(data_case["Y"], dtype=np.float32)

    x = make_tensor_value_info("x", TensorProto.FLOAT,  input.shape)
    y = make_tensor_value_info("y", TensorProto.FLOAT,  output.shape)

    k_ = make_tensor_value_info("k", TensorProto.INT64, [1])

    node = onnx.helper.make_node(
        "Trilu",
        inputs=["x","k"],
        outputs=["y"],
        upper=upper,
    )


    graph = make_graph(
        [node],
        "trilu_graph",
        [x, k_],
        [y],
    )

    model = make_model(graph, producer_name="trilu_tril", ir_version=10)
    onnx.save_model(model, os.path.join("models",f"trilu_{lab}.onnx"))

    # save x and y as npy
    np.save(os.path.join("data",f"input_trilu_{lab}_0.npy"), input.data)
    np.save(os.path.join("data",f"input_trilu_{lab}_1.npy"), np.array([k], dtype=np.int64).data)

    np.save(os.path.join("data",f"output_trilu_{lab}.npy"), output.data)
