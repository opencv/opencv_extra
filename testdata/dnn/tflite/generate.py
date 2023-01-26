# Use this script to generate test data for dnn module and TFLite models
import os
import numpy as np
import tensorflow as tf
import mediapipe as mp

import cv2 as cv

testdata = os.environ['OPENCV_TEST_DATA_PATH']

image = cv.imread(os.path.join(testdata, "cv", "shared", "lena.png"))
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

def run_tflite_model(model_name, inp_size):
    interpreter = tf.lite.Interpreter(model_name + ".tflite",
                                      experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run model
    inp = cv.resize(image, inp_size)
    inp = np.expand_dims(inp, 0)
    inp = inp.astype(np.float32) / 255  # NHWC

    interpreter.set_tensor(input_details[0]['index'], inp)

    interpreter.invoke()

    for details in output_details:
        out = interpreter.get_tensor(details['index'])  # Or use an intermediate layer index
        out_name = details['name']
        np.save(f"{model_name}_out_{out_name}.npy", out)


def run_mediapipe_solution(solution, inp_size):
    with solution as selfie_segmentation:
        inp = cv.resize(image, inp_size)
        results = selfie_segmentation.process(inp)
        np.save(f"selfie_segmentation_out_activation_10.npy", results.segmentation_mask)


# # There is no solution class for Hair Segmentation from Mediapipe.
# # So we have to run it as a generic model.
# def run_hair_segmentation():
#     config_text = """
#     input_stream: "FLOATS:input"
#     output_stream: "TENSORS:output"

#     node {
#         calculator: "TfLiteCustomOpResolverCalculator"
#         output_side_packet: "OP_RESOLVER:op_resolver"
#     }

#     node {
#         calculator: "ImageToTensorCalculator"
#         input_stream: "IMAGE:input"
#         output_stream: "TENSORS:input_tensor"
#         options: {
#             [mediapipe.ImageToTensorCalculatorOptions.ext] {
#                 output_tensor_width: 512
#                 output_tensor_height: 512
#                 output_tensor_float_range {
#                     min: 0.0
#                     max: 1.0
#                 }
#             }
#         }
#     }

#     node {
#         calculator: "InferenceCalculator"
#         input_stream: "TENSORS:input_tensor"
#         output_stream: "TENSORS:output_tensor"
#         input_side_packet: "OP_RESOLVER:op_resolver"
#         options {
#             [mediapipe.InferenceCalculatorOptions.ext] {
#                 model_path: "hair_segmentation.tflite"
#             }
#         }
#     }

#     node {
#         calculator: "TensorsToFloatsCalculator"
#         input_stream: "TENSORS:output_tensor"
#         output_stream: "FLOATS:output"
#     }
#     """

#     image = cv.imread(os.path.join(testdata, "cv", "shared", "lena.png"))
#     inp = cv.cvtColor(image, cv.COLOR_BGR2RGBA)

#     graph = mp.CalculatorGraph(graph_config=config_text)

#     output_packets = []

#     graph.observe_output_stream(
#         'output',
#         lambda stream_name, packet:
#             output_packets.append(mp.packet_getter.get_float_list(packet)))

#     num_iters = 1
#     mask = np.zeros((512, 512), dtype=np.uint8)
#     for i in range(num_iters):
#         inp[:, :, -1] = mask  # A hair mask from previous frame (or from previous iteration for image)

#         graph.start_run()

#         graph.add_packet_to_input_stream(
#             'input',
#             mp.packet_creator.create_image_frame(image_format=mp.ImageFormat.SRGBA,
#                                                  data=inp).at(1))
#         graph.close()

#         print(len(output_packets))
#         print(len(output_packets[-1]))
#         mask = np.array(output_packets[-1]).reshape(512, 512, 2).astype(np.float32)
#         # mask = np.argmax(mask, axis=-1)
#         print(np.min(mask[:, :, 0]), np.max(mask[:, :, 0]))
#         mask = (mask[:, :, 0] - np.min(mask[:, :, 0])) / (np.max(mask[:, :, 0]) - np.min(mask[:, :, 0]))
#         print(np.min(mask), np.max(mask))
#         # mask = (mask == 1) * 255
#         cv.imwrite(f"mask_{i}.png", (mask * 255).astype(np.uint8))

#     # np.save("hair_segmentation_out_conv2d_transpose_4.npy", mask)


run_tflite_model("face_landmark", (192, 192))
run_tflite_model("face_detection_short_range", (128, 128))

run_mediapipe_solution(mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0), (256, 256))
