import os
import numpy as np
import onnx
import onnxscript as ost
from onnxscript import opset19 as op # opset19 is the lastest by 202309
from onnxscript import opset11
from onnxscript import opset13

###############
### CAUTION!!!
### Be sure to put random-generated constant numpy arrays out of @ost.script() decorated fucntion.
### Otherwise random values change each time eager mode is entereded.
### See discussions in https://github.com/microsoft/onnxscript/issues/1313
###############

np.random.seed(0)

def make_model_and_data(model, *args, **kwargs):
    name = model._name

    # TODO: support multiple outputs
    outputs = model(*args) # eager mode

    # Save model
    model_proto = model.to_model_proto()
    try:
        onnx.checker.check_model(model_proto)
    except onnx.checker.ValidationError as e:
        print(f"Model {name} is invalid: {e}. Skipping ...")
        return False
    else:
        save_path = "./models/{}.onnx".format(name)
        print(f"Model {name} is valid! Saved to {save_path}")
        model_proto_ = onnx.shape_inference.infer_shapes(model_proto)
        onnx.save(model_proto_, save_path)

    # Save inputs
    inputs = args
    if "force_saving_input_as_dtype_float32" in kwargs and kwargs["force_saving_input_as_dtype_float32"]:
        inputs = []
        for input in args:
            inputs.append(input.astype(np.float32))
    if len(args) == 1:
        input_file = os.path.join("data", "input_" + name)
        if "save_inputs_as_pb" in kwargs and kwargs["save_inputs_as_pb"]:
            input_tensor = onnx.numpy_helper.from_array(inputs[0])
            onnx.save_tensor(input_tensor, input_file + ".pb")
        else:
            np.save(input_file, inputs[0])
    else:
        for idx, input in enumerate(inputs, start=0):
            input_file = os.path.join("data", "input_{}_{}".format(name, idx))
            if "save_inputs_as_pb" in kwargs and kwargs["save_inputs_as_pb"]:
                input_tensor = onnx.numpy_helper.from_array(input)
                onnx.save_tensor(input_tensor, input_file + ".pb")
            else:
                np.save(input_file, input)

    # Save outputs
    if isinstance(outputs, tuple):
        for idx, output in enumerate(outputs):
            output_filepath = os.path.join("data", "output_{}_{}".format(name, idx))
            if "force_saving_output_as_dtype_float32" in kwargs and kwargs["force_saving_output_as_dtype_float32"]:
                output = output.astype(np.float32)
            if "save_outputs_as_pb" in kwargs and kwargs["save_outputs_as_pb"]:
                input_tensor = onnx.numpy_helper.from_array(output)
                onnx.save_tensor(input_tensor, output_filepath + ".pb")
            else:
                np.save(output_filepath, output)
    else:
        output = outputs
        if "force_saving_output_as_dtype_float32" in kwargs and kwargs["force_saving_output_as_dtype_float32"]:
            output = output.astype(np.float32)
        output_filepath = os.path.join("data", "output_" + name)
        if "save_outputs_as_pb" in kwargs and kwargs["save_outputs_as_pb"]:
            output_tensor = onnx.numpy_helper.from_array(output)
            onnx.save_tensor(output_tensor, output_filepath + ".pb")
        else:
            np.save(output_filepath, output)

'''
    It builds a model with two Gather ops sharing a single same indices:

    [Input] -> Gather(indices=0) -> Gather(indices=0) -> [Output]

    , where the two indices constants have the same name.
'''
@ost.script()
def gather_shared_indices(x: ost.FLOAT[2, 1, 3, 4]) -> ost.FLOAT[3, 4]:
    indices = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([0], dtype=np.int64)))
    y0 = op.Gather(x, indices, axis=0)
    y1 = op.Gather(y0, indices, axis=0)
    return y1
make_model_and_data(gather_shared_indices, np.random.rand(2, 1, 3, 4).astype(np.float32))

'''
    [Input] -> Greater(B=61) -> [Output]
                        \
                        dtype=np.int64
'''
@ost.script()
def greater_input_dtype_int64(x: ost.FLOAT[27, 9]) ->ost.BOOL[27, 9]:
    y = op.Greater(x, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([61], dtype=np.int64))))
    return y
make_model_and_data(greater_input_dtype_int64, np.random.randint(0, 100, size=[27, 9], dtype=np.int64), force_saving_input_as_dtype_float32=True, force_saving_output_as_dtype_float32=True)

@ost.script()
def two_resizes_with_shared_subgraphs(x: ost.FLOAT["batch", 1, "height", "width"], y: ost.FLOAT[1, 1, 3, 2], z: ost.FLOAT[1, 1, 2, 1]) ->ost.FLOAT["batch", 1, "height", "width"]:
    shape_src_1 = opset11.Shape(x)
    shape_src_2 = opset11.Shape(x)
    gather_h = opset11.Gather(shape_src_1, opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([2], dtype=np.int64))), axis=0)
    gather_w = opset11.Gather(shape_src_2, opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([3], dtype=np.int64))), axis=0)
    unsqueeze_w_1 = opset11.Unsqueeze(gather_w, axes=[0])
    unsqueeze_w_2 = opset11.Unsqueeze(gather_w, axes=[0])
    unsqueeze_h_1 = opset11.Unsqueeze(gather_h, axes=[0])
    unsqueeze_h_2 = opset11.Unsqueeze(gather_h, axes=[0])
    concat_1 = opset11.Cast(opset11.Concat(unsqueeze_h_1, unsqueeze_w_1, axis=0), to=ost.INT64.dtype)
    concat_2 = opset11.Cast(opset11.Concat(unsqueeze_h_2, unsqueeze_w_2, axis=0), to=ost.INT64.dtype)

    # This op is required to test double node removal
    y = opset11.Add(y, opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [1], np.array([0.5], dtype=np.float32))))

    # First branch
    sliced = opset11.Slice(opset11.Shape(y),
        starts=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64))),
        ends=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([2], dtype=np.int64))),
        axes=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64))),
    )
    concat_y = opset11.Concat(sliced, concat_1, axis=0)
    resized_y = opset11.Resize(y,
        roi=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [0], np.empty([0]))),
        scales=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [0], np.empty([0]))),
        sizes=concat_y,
        coordinate_transformation_mode='pytorch_half_pixel',
        cubic_coeff_a=-0.75,
        mode='linear',
        nearest_mode='floor'
    )

    # Second branch
    sliced = opset11.Slice(opset11.Shape(z),
        starts=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64))),
        ends=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([2], dtype=np.int64))),
        axes=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64))),
    )
    concat_z = opset11.Concat(sliced, concat_2, axis=0)
    resized_z = opset11.Resize(z,
        roi=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [0], np.empty([0]))),
        scales=opset11.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [0], np.empty([0]))),
        sizes=concat_z,
        coordinate_transformation_mode='pytorch_half_pixel',
        cubic_coeff_a=-0.75,
        mode='linear',
        nearest_mode='floor'
    )

    return opset11.Add(resized_y, resized_z)

make_model_and_data(two_resizes_with_shared_subgraphs, np.random.rand(1, 1, 4, 5).astype(np.float32), np.random.rand(1, 1, 3, 2).astype(np.float32), np.random.rand(1, 1, 2, 1).astype(np.float32))


@ost.script()
def bias_gelu(x: ost.FLOAT[1, 2, 3]) -> ost.FLOAT[1, 2, 3]:
    bias = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [3], np.array([0.1, 0.3, 0.2], dtype=np.float32)))
    add1 = op.Add(x, bias)
    tmp = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([np.sqrt(2)], dtype=np.float32)))
    div = op.Div(add1, tmp)
    erf = op.Erf(div)
    tmp_0 = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([1], dtype=np.float32)))
    add2 = op.Add(erf, tmp_0)
    mul = op.Mul(add1, add2)
    tmp_1 = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([0.5], dtype=np.float32)))
    return op.Mul(mul, tmp_1)

make_model_and_data(bias_gelu, np.random.rand(1, 2, 3).astype(np.float32))

batch_size = 1
sequence_length = 320
input_hidden_size = 48
qk_hidden_size = 48
v_hidden_size = 48
num_heads = 4
qk_head_size = int(qk_hidden_size / num_heads)
v_head_size = int(v_hidden_size / num_heads)
attention_weight = np.random.rand(input_hidden_size, qk_hidden_size + qk_hidden_size + v_hidden_size).astype(np.float32)
attention_bias = np.random.rand(qk_hidden_size + qk_hidden_size + v_hidden_size).astype(np.float32)

'''
    Attention Subgraph.

                   [Input](BxSxW)
                      |
                   LayerNorm
                      |
                   Transpose(perm=[1, 0, 2])
                      |
                      | (SxBxW)
                      |
                    Matmul[Weight(Wx3W)]
                      |
                     Add[Bias(3W)]
          /           |           \
      q_Slice      k_Slice      v_Slice   (output(SxBxW))
         |            |            |
     q_Reshape    k_Reshape    v_Reshape  (output(Sx(BxN)xH), could be optional if N=1)
         |            |            |
    q_Transpose  k_Transpose  v_Transpose
      (1,0,2)      (1,2,0)    (perm=1,0,2)
         |((BxN)xSxH) |((BxN)xHxS) |
       q_Div         /            /
         \          /            /
          qk_MatMul             /
              |                /
         qk_Softmax           /
              | ((BxN)xSxS)  / ((BxN)xSxH)
               \            /
                 qkv_MatMul  (output((BxN)xSxH))
                     |
                 Transpose(perm=1,2,0)
                     |
                  Reshape  (output(SxH))
                     |
                   MatMul
                     |
                    Add
                     |
                  [Output](BxSxW)
'''

@ost.script()
def attention(x: ost.FLOAT[batch_size, sequence_length, input_hidden_size]) -> ost.FLOAT[batch_size, sequence_length, input_hidden_size]:
    transpose = op.Transpose(x, perm=[1, 0, 2])
    qkv_matmul_weight = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, attention_weight.shape, attention_weight))
    qkv_matmul = op.MatMul(transpose, qkv_matmul_weight)

    qkv_add_bias = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, attention_bias.shape, attention_bias))
    qkv_add = op.Add(qkv_add_bias, qkv_matmul)

    # q path
    q_path_slice = op.Slice(qkv_add,
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([-1], dtype=np.int64))))
    q_path_reshape = op.Reshape(q_path_slice, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([sequence_length, batch_size * num_heads, qk_head_size], dtype=np.int64))), allowzero=0)
    q_path_transpose = op.Transpose(q_path_reshape, perm=[1, 0, 2])
    q_path_div = op.Div(q_path_transpose, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([np.sqrt(qk_hidden_size)], dtype=np.float32))))
    # k path
    k_path_slice = op.Slice(qkv_add,
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size + qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([-1], dtype=np.int64))))
    k_path_reshape = op.Reshape(k_path_slice, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([sequence_length, batch_size * num_heads, qk_head_size], dtype=np.int64))), allowzero=0)
    k_path_transpose = op.Transpose(k_path_reshape, perm=[1, 2, 0])

    # qk path
    qk_matmul = op.MatMul(q_path_div, k_path_transpose)
    qk_softmax = op.Softmax(qk_matmul, axis=-1)

    # v path
    v_path_slice = op.Slice(qkv_add,
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size + qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size + qk_hidden_size + v_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([-1], dtype=np.int64))))
    v_path_reshape = op.Reshape(v_path_slice, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([sequence_length, batch_size * num_heads, v_head_size], dtype=np.int64))), allowzero=0)
    v_path_transpose = op.Transpose(v_path_reshape, perm=[1, 0, 2])

    # matmul
    matmul = op.MatMul(qk_softmax, v_path_transpose)
    trans = op.Transpose(matmul, perm=[1, 0, 2])
    reshape = op.Reshape(trans, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([batch_size, sequence_length, v_hidden_size], dtype=np.int64))))

    return reshape

make_model_and_data(attention, np.random.rand(batch_size, sequence_length, input_hidden_size).astype(np.float32))

batch_size = 1
sequence_length = 320
input_hidden_size = 48
qk_hidden_size = 48
v_hidden_size = 48
num_heads = 1
qk_head_size = int(qk_hidden_size / num_heads)
v_head_size = int(v_hidden_size / num_heads)
attention_weight = np.random.rand(input_hidden_size, qk_hidden_size + qk_hidden_size + v_hidden_size).astype(np.float32)
attention_bias = np.random.rand(qk_hidden_size + qk_hidden_size + v_hidden_size).astype(np.float32)

'''
    Single-head attention subgraph like the above one but without the appended Reshape after each Slice.
    Also v_Slice.end = INT64_MAX which stands for slicing till the end of dimension of the actual tensor.
'''

@ost.script()
def attention_single_head(x: ost.FLOAT[batch_size, sequence_length, input_hidden_size]) -> ost.FLOAT[batch_size, sequence_length, input_hidden_size]:
    transpose = op.Transpose(x, perm=[1, 0, 2])
    qkv_matmul_weight = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, attention_weight.shape, attention_weight))
    qkv_matmul = op.MatMul(transpose, qkv_matmul_weight)

    qkv_add_bias = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, attention_bias.shape, attention_bias))
    qkv_add = op.Add(qkv_add_bias, qkv_matmul)

    # q path
    q_path_slice = op.Slice(qkv_add,
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([0], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([-1], dtype=np.int64))))
    q_path_transpose = op.Transpose(q_path_slice, perm=[1, 0, 2])
    q_path_div = op.Div(q_path_transpose, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([np.sqrt(qk_hidden_size)], dtype=np.float32))))
    # k path
    k_path_slice = op.Slice(qkv_add,
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size + qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([-1], dtype=np.int64))))
    k_path_transpose = op.Transpose(k_path_slice, perm=[1, 2, 0])

    # qk path
    qk_matmul = op.MatMul(q_path_div, k_path_transpose)
    qk_softmax = op.Softmax(qk_matmul, axis=-1)

    # v path
    v_path_slice = op.Slice(qkv_add,
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([qk_hidden_size + qk_hidden_size], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([np.iinfo(np.int64).max], dtype=np.int64))),
                        op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], np.array([-1], dtype=np.int64))))
    v_path_transpose = op.Transpose(v_path_slice, perm=[1, 0, 2])

    # matmul
    matmul = op.MatMul(qk_softmax, v_path_transpose)
    trans = op.Transpose(matmul, perm=[1, 0, 2])
    reshape = op.Reshape(trans, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [3], np.array([batch_size, sequence_length, v_hidden_size], dtype=np.int64))))

    return reshape

make_model_and_data(attention_single_head, np.random.rand(batch_size, sequence_length, input_hidden_size).astype(np.float32))

# Einsum_const_inputs

input_0_data = np.random.rand(3, 2, 2, 4).astype(np.float32)
input_1_data = np.random.rand(2, 2, 4).astype(np.float32)

@ost.script()
def einsum_const_inputs(input_0: ost.FLOAT[3, 2, 2, 4]) -> ost.FLOAT[3, 2, 2, 2]:
    input_1 = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, input_1_data.shape, input_1_data))
    return op.Einsum(input_0, input_1, equation="bhwc, hkc -> bhwk")

make_model_and_data(einsum_const_inputs, input_0_data)

''' This subgraph looks the same as LayerNorm expanded, but it has
    axes=1 in ReduceMean which does not meet the requirement of LayerNorm:
        - axes[-1] = -1 or the axis of last dimension
        - adjacent axes, e.g. [1, 2, 3] or [-3, -2, -1]
'''

n = 1
c = 4
h = w = 8
mul_weight = np.random.rand(c, 1, 1).astype(np.float32)
add_weight = np.random.rand(c, 1, 1).astype(np.float32)

@ost.script()
def layer_norm_no_fusion(x: ost.FLOAT[n, c, h, w]) -> ost.FLOAT[n, c, h, w]:
    reduce_mean = opset13.ReduceMean(x, axes=[1], keepdims=1)
    sub = opset13.Sub(x, reduce_mean)

    pow = opset13.Pow(sub, opset13.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([2], dtype=np.float32))))
    reduce_mean_1 = opset13.ReduceMean(pow, axes=[1], keepdims=1)
    add = opset13.Add(reduce_mean_1, opset13.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([9.999999974752427e-7], dtype=np.float32))))
    sqrt = opset13.Sqrt(add)

    div = opset13.Div(sub, sqrt)
    mul = opset13.Mul(opset13.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [c, 1, 1], mul_weight)), div)
    add = opset13.Add(mul, opset13.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [c, 1, 1], add_weight)))

    return add
make_model_and_data(layer_norm_no_fusion, np.random.rand(n, c, h, w).astype(np.float32))


''' Subgraph: [Input] -> MatMul<B> -> Add<A> -> [Output]
'''

b = 2
m = 32
n = 64
k = 16
weight_data = np.random.rand(k, n).astype(np.float32)
bias_data = np.random.rand(n).astype(np.float32)

@ost.script()
def biased_matmul(x: ost.FLOAT[b, m, k]) -> ost.FLOAT[b, m, n]:
    weight = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [k, n], weight_data))
    matmul = op.MatMul(x, weight)
    bias = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [n], bias_data))
    return op.Add(bias, matmul)
make_model_and_data(biased_matmul, np.random.rand(b, m, k).astype(np.float32))

''' Subgraph: [Input] -> Clip<min=0, max=6> -> Add<B=6> -> Clip<min=0, max=6> -> Add<B=6> -> [Output]

    Here max=6 and B=6 shares the same Constant node.
'''

@ost.script()
def clip_div_shared_constant(x: ost.FLOAT[1, 8, 12, 10]) -> ost.FLOAT[1, 8, 12, 10]:
    Constant_output_0 = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([0], dtype=np.float32)))
    Constant_1_output_0 = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.FLOAT, [], np.array([6], dtype=np.float32)))

    div = op.Div(x, Constant_1_output_0)
    clip = op.Clip(div, Constant_output_0, Constant_1_output_0)
    return clip
make_model_and_data(clip_div_shared_constant, np.random.rand(1, 8, 12, 10).astype(np.float32))

''' Subgraph [Input] -> MatMul<B> -> [Outpunt]
'''

B = np.random.randn(16, 8).astype(np.float32)

@ost.script()
def matmul_bcast(x: ost.FLOAT[64, 1, 16]) -> ost.FLOAT[64, 1, 8]:
    return op.MatMul(x, op.Constant(value=onnx.numpy_helper.from_array(B)))
make_model_and_data(matmul_bcast, np.random.randn(64, 1, 16).astype(np.float32))

''' TopK conformance
'''

top_k_K_arr = np.array([3], dtype=np.int64)
@ost.script()
def top_k(x: ost.FLOAT[3, 4]) -> (ost.FLOAT[3, 3], ost.INT64[3, 3]):
    values, indices = op.TopK(x, op.Constant(value=onnx.numpy_helper.from_array(top_k_K_arr)), axis=1)
    return values, indices

@ost.script()
def top_k_negative_axis(x: ost.FLOAT[3, 4]) -> (ost.FLOAT[3, 3], ost.INT64[3, 3]):
    values, indices = op.TopK(x, op.Constant(value=onnx.numpy_helper.from_array(top_k_K_arr)), axis=-1)
    return values, indices

@ost.script()
def top_k_smallest(x: ost.FLOAT[3, 4]) -> (ost.FLOAT[3, 3], ost.INT64[3, 3]):
    values, indices = op.TopK(x, op.Constant(value=onnx.numpy_helper.from_array(top_k_K_arr)), axis=1, largest=0, sorted=1)
    return values, indices

top_k_input0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
top_k_input1 = [0, 1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 8]
top_k_input0_arr = np.array(top_k_input0, dtype=np.float32).reshape(3, 4)
top_k_input1_arr = np.array(top_k_input1, dtype=np.float32).reshape(3, 4)
make_model_and_data(top_k,               top_k_input0_arr, save_inputs_as_pb=True, save_outputs_as_pb=True)
make_model_and_data(top_k_negative_axis, top_k_input0_arr, save_inputs_as_pb=True, save_outputs_as_pb=True)
make_model_and_data(top_k_smallest,      top_k_input1_arr, save_inputs_as_pb=True, save_outputs_as_pb=True)
