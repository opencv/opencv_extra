# Use this script to generate test data for dnn module and TFLite models
import os
import numpy as np
import tensorflow as tf
import mediapipe as mp

import cv2 as cv

testdata = os.environ['OPENCV_TEST_DATA_PATH']

image = cv.imread(os.path.join(testdata, "cv", "shared", "lena.png"))
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

def run_tflite_model(model_name, inp_size=None, inp=None):
    interpreter = tf.lite.Interpreter(model_name + ".tflite",
                                      experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run model
    if inp is None:
        inp = cv.resize(image, inp_size)
        inp = np.expand_dims(inp, 0)
        inp = inp.astype(np.float32) / 255  # NHWC

    interpreter.set_tensor(input_details[0]['index'], inp)

    interpreter.invoke()

    for details in output_details:
        out = interpreter.get_tensor(details['index'])  # Or use an intermediate layer index
        out_name = details['name']
        np.save(f"{model_name}_out_{out_name.replace(":", "_")}.npy", out)


def run_mediapipe_solution(solution, inp_size):
    with solution as selfie_segmentation:
        inp = cv.resize(image, inp_size)
        results = selfie_segmentation.process(inp)
        np.save(f"selfie_segmentation_out_activation_10.npy", results.segmentation_mask)

run_tflite_model("face_landmark", (192, 192))
run_tflite_model("face_detection_short_range", (128, 128))

# Download from https://storage.googleapis.com/mediapipe-assets/facemesh2_lite_iris_faceflag_2023_02_14.tflite?generation=1681322470818178
# run_tflite_model("facemesh2_lite_iris_faceflag_2023_02_14", (192, 192))

# source: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
face_blendshapes_inp = np.load("facemesh2_lite_iris_faceflag_2023_02_14_out_StatefulPartitionedCall:1.npy").reshape(-1, 3)
face_blendshapes_inp = face_blendshapes_inp[[
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157,
    158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282,
    283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356,
    361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405,
    409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
]]
face_blendshapes_inp = face_blendshapes_inp[:, [0, 1]].reshape(1, -1, 2)
np.save("face_blendshapes_inp.npy", np.ascontiguousarray(face_blendshapes_inp))
run_tflite_model("face_blendshapes", inp=face_blendshapes_inp)

run_mediapipe_solution(mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0), (256, 256))

# Save TensorFlow model as TFLite
def save_tflite_model(model, inp, name):
    func = model.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    tflite_model = converter.convert()

    with open(f'{name}.tflite', 'wb') as f:
        f.write(tflite_model)

    out = model(inp)
    out = np.array(out)

    # convert NHWC to NCHW format
    if inp.ndim == 4:
        inp = inp.transpose(0, 3, 1, 2)
        inp = np.copy(inp, order='C').astype(inp.dtype)

    if out.ndim == 4:
        out = out.transpose(0, 3, 1, 2)
        out = np.copy(out, order='C').astype(out.dtype)

    np.save(f'{name}_inp.npy', inp)
    np.save(f'{name}_out_Identity.npy', out)


@tf.function(input_signature=[tf.TensorSpec(shape=[1, 3, 3, 1], dtype=tf.float32)])
def replicate_by_pack(x):
    pack_1 = tf.stack([x, x], axis=3)
    reshape_1 = tf.reshape(pack_1, [1, 3, 6, 1])
    pack_2 = tf.stack([reshape_1, reshape_1], axis=2)
    reshape_2 = tf.reshape(pack_2, [1, 6, 6, 1])
    scaled = tf.image.resize(reshape_2, size=(3, 3), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return scaled + x

inp = np.random.standard_normal((1, 3, 3, 1)).astype(np.float32)
save_tflite_model(replicate_by_pack, inp, 'replicate_by_pack')

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 3], dtype=tf.float32)])
def split(x):
    splitted = tf.split(
        x, 3, axis=-1, num=None, name='split'
    )
    return tf.concat((splitted[2], splitted[1], splitted[0]), axis=-1)

inp = np.random.standard_normal((1, 3)).astype(np.float32)
save_tflite_model(split, inp, 'split')

def keras_to_tf(model, input_shape):
    tf_func = tf.function(
      model.call,
      input_signature=[tf.TensorSpec(input_shape, tf.float32)],
    )
    inp = np.random.standard_normal((input_shape)).astype(np.float32)

    return tf_func, inp

fully_connected = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3),
  tf.keras.layers.ReLU(),
  tf.keras.layers.Softmax(),
])

fully_connected, inp = keras_to_tf(fully_connected, (1, 2))
save_tflite_model(fully_connected, inp, 'fully_connected')

permutation_3d = tf.keras.models.Sequential([
  tf.keras.layers.Permute((2, 1))
])

permutation_3d, inp = keras_to_tf(permutation_3d, (1, 2, 3))
save_tflite_model(permutation_3d, inp, 'permutation_3d')

# (1, 2, 3) is temporarily disabled as TFLiteConverter produces a incorrect graph in this case
permutation_4d_list = [(1, 3, 2), (2, 1, 3), (2, 3, 1)]
for perm_axis in permutation_4d_list:
    permutation_4d_model = tf.keras.models.Sequential([
        tf.keras.layers.Permute(perm_axis),
        tf.keras.layers.Conv2D(3, 1)
    ])

    permutation_4d_model, inp = keras_to_tf(permutation_4d_model, (1, 2, 3, 4))
    model_name = f"permutation_4d_0{''.join(map(str, perm_axis))}"
    save_tflite_model(permutation_4d_model, inp, model_name)

global_average_pooling_2d = tf.keras.models.Sequential([
  tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
  tf.keras.layers.ZeroPadding2D(1),
  tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
])

global_average_pooling_2d, inp = keras_to_tf(global_average_pooling_2d, (1, 7, 7, 5))
save_tflite_model(global_average_pooling_2d, inp, 'global_average_pooling_2d')

global_max_pool = tf.keras.models.Sequential([
  tf.keras.layers.GlobalMaxPool2D(keepdims=True),
  tf.keras.layers.ZeroPadding2D(1),
  tf.keras.layers.GlobalMaxPool2D(keepdims=True)
])

global_max_pool, inp = keras_to_tf(global_max_pool, (1, 7, 7, 5))
save_tflite_model(global_max_pool, inp, 'global_max_pooling_2d')

leakyRelu = tf.keras.models.Sequential([
    tf.keras.layers.LeakyReLU()
])

leakyRelu, inp = keras_to_tf(leakyRelu, (1, 7, 7, 5))
save_tflite_model(leakyRelu, inp, 'leakyRelu')

@tf.function(input_signature=[tf.TensorSpec(shape=[2, 1, 1, 4], dtype=tf.float32)])
def strided_slice(x):
    return x[-1:, ..., ::2]

inp = np.random.standard_normal((2, 1, 1, 4)).astype(np.float32)
save_tflite_model(strided_slice, inp, 'strided_slice')
