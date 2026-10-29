"""
Regenerate the hand_landmark_lite reference outputs.

The input must match Test_TFLite.hand_landmark exactly:
    Mat hand = imread("dnn/pose.png")(Rect(0, 138, 140, 140));
    blobFromImage(hand, 1.0 / 255, Size(224, 224), Scalar(), true, false);

hand_landmark_lite.tflite is not stored in git, run download_models.py first.
"""
import os

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

TFLITE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.environ.get("OPENCV_TEST_DATA_PATH",
                          os.path.normpath(os.path.join(TFLITE, "..", "..")))

NAME = "hand_landmark_lite"
MODEL = os.path.join(TFLITE, NAME + ".tflite")
IMAGE = os.path.join(TESTDATA, "dnn", "pose.png")

for path in (MODEL, IMAGE):
    if not os.path.exists(path):
        raise SystemExit("not found: " + path)

image = cv2.imread(IMAGE)
hand = cv2.cvtColor(image[138:138 + 140, 0:0 + 140], cv2.COLOR_BGR2RGB)
inp = np.expand_dims(cv2.resize(hand, (224, 224)), 0).astype(np.float32) / 255

interp = Interpreter(model_path=MODEL)
interp.allocate_tensors()
interp.set_tensor(interp.get_input_details()[0]["index"], inp)
interp.invoke()

for det in interp.get_output_details():
    out = interp.get_tensor(det["index"])
    path = os.path.join(TFLITE, "%s_out_%s.npy" % (NAME, det["name"].replace(":", "_")))
    np.save(path, out)
    print("saved %s  shape=%s" % (os.path.basename(path), out.shape))
