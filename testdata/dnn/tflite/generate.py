# Use this script to generate test data for dnn module and TFLite models
import os
import numpy as np
import tensorflow as tf

import cv2 as cv

testdata = os.environ['OPENCV_TEST_DATA_PATH']

interpreter = tf.lite.Interpreter("face_landmark.tflite",
                                  experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run model
image = cv.imread(os.path.join(testdata, "cv", "shared", "lena.png"))
inp = cv.resize(image, (192, 192))
inp = cv.cvtColor(inp, cv.COLOR_BGR2RGB)
inp = np.expand_dims(inp, 0)
inp = inp.astype(np.float32) / 255  # NHWC

interpreter.set_tensor(input_details[0]['index'], inp)

interpreter.invoke()

out = interpreter.get_tensor(output_details[0]['index'])  # Or use an intermediate layer index
print(out.shape)
np.save("face_landmark_out.npy", out)
