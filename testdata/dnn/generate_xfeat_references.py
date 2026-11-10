#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
"""Generate the XFeat reference NPY files used by the Features2d_XFeat regression tests.

The dense maps come from testdata/dnn/onnx/models/xfeat.onnx run with ONNX Runtime CPU on the
same letterboxed input cv::XFeat builds. Keypoints, scores and descriptors are then computed by
the reference XFeat.detectAndCompute (https://github.com/verlab/accelerated_features), so the
files pin the reference sampling: nearest heatmap and bilinear reliability lookups, and bicubic
descriptors from the per-pixel L2-normalised descriptor map, all through grid_sample with
align_corners=False.

Dependencies: numpy, opencv-python, onnxruntime, torch, tqdm, and a checkout of
accelerated_features. Generated with accelerated_features e92685f, onnxruntime 1.31.1,
opencv-python 5.0.0 and torch 2.15.0:

    python generate_xfeat_references.py --testdata <opencv_extra>/testdata \
        --images <opencv_extra>/testdata/cv/shared --xfeat_repo <accelerated_features>
"""

import argparse
import hashlib
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch


INPUT_SIZE = 640
IMAGES = {"box": "box.png", "box_in_scene": "box_in_scene.png"}


def letterbox(gray):
    # Same arithmetic as blobFromImageWithParams with DNN_PMODE_LETTERBOX and INTER_LINEAR_EXACT.
    height, width = gray.shape
    factor = min(np.float32(INPUT_SIZE) / np.float32(width), np.float32(INPUT_SIZE) / np.float32(height))
    resized_w = int(np.float32(width) * factor)
    resized_h = int(np.float32(height) * factor)
    resized = cv.resize(gray, (resized_w, resized_h), interpolation=cv.INTER_LINEAR_EXACT)
    top = (INPUT_SIZE - resized_h) // 2
    left = (INPUT_SIZE - resized_w) // 2
    padded = cv.copyMakeBorder(resized, top, INPUT_SIZE - top - resized_h, left,
                               INPUT_SIZE - left - resized_w, cv.BORDER_CONSTANT, value=0)
    blob = padded.astype(np.float32) * np.float32(1.0 / 255.0)
    return blob[None, None], (resized_w, resized_h), (left, top)


def make_reference(xfeat_repo, session):
    sys.path.insert(0, str(xfeat_repo))
    from modules.xfeat import XFeat

    class OnnxXFeatModel(torch.nn.Module):
        """XFeatModel.forward served by the ONNX export, NHWC outputs returned as NCHW."""

        def forward(self, x):
            feats, keypoints, heatmap = session.run(
                ["output_feats", "output_keypoints", "output_heatmap"], {"input": x.cpu().numpy()})
            return (torch.from_numpy(feats).permute(0, 3, 1, 2),
                    torch.from_numpy(keypoints).permute(0, 3, 1, 2),
                    torch.from_numpy(heatmap))

    xfeat = XFeat(weights=None)
    xfeat.net = OnnxXFeatModel()
    return xfeat


def extract(xfeat, image, threshold):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blob, (resized_w, resized_h), (left, top) = letterbox(gray)
    out = xfeat.detectAndCompute(torch.from_numpy(blob), top_k=INPUT_SIZE * INPUT_SIZE,
                                 detection_threshold=threshold)[0]
    padded = out["keypoints"].numpy().astype(np.float32)
    scores = out["scores"].numpy().astype(np.float32)
    descriptors = out["descriptors"].numpy().astype(np.float32)

    # Map back to image pixels like cv::XFeat and drop keypoints whose pixel is in the padding.
    height, width = gray.shape
    inv = np.array([width / np.float32(resized_w), height / np.float32(resized_h)], dtype=np.float32)
    pixels = (padded - np.array([left, top], dtype=np.float32) + np.float32(0.5)) * inv - np.float32(0.5)
    rounded = np.rint(pixels)
    inside = ((rounded[:, 0] >= 0) & (rounded[:, 1] >= 0) &
              (rounded[:, 0] < width) & (rounded[:, 1] < height))
    pixels, scores, descriptors = pixels[inside], scores[inside], descriptors[inside]

    # Raster order of the padded position: the order cv::XFeat emits keypoints in.
    order = np.lexsort((padded[inside, 0], padded[inside, 1]))
    keypoints = np.concatenate([pixels, scores[:, None]], axis=1)[order]
    return keypoints, descriptors[order]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--testdata", type=Path, required=True, help="opencv_extra/testdata")
    parser.add_argument("--images", type=Path, required=True, help="opencv/samples/data")
    parser.add_argument("--xfeat_repo", type=Path, required=True, help="accelerated_features checkout")
    parser.add_argument("--threshold", type=float, default=0.5, help="cv::XFeat scoreThreshold")
    parser.add_argument("--output", type=Path, help="output directory (default: testdata/dnn)")
    args = parser.parse_args()

    model = args.testdata / "dnn" / "onnx" / "models" / "xfeat.onnx"
    print("Model: {} (SHA-1 {})".format(model, hashlib.sha1(model.read_bytes()).hexdigest()))
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    xfeat = make_reference(args.xfeat_repo, session)
    output = args.output or args.testdata / "dnn"
    output.mkdir(parents=True, exist_ok=True)

    for tag, name in IMAGES.items():
        image = cv.imread(str(args.images / name), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(args.images / name)
        keypoints, descriptors = extract(xfeat, image, args.threshold)
        np.save(output / "xfeat_{}_640_kpts.npy".format(tag), keypoints)
        np.save(output / "xfeat_{}_640_desc.npy".format(tag), descriptors)
        print("{}: {} keypoints".format(tag, len(keypoints)))


if __name__ == "__main__":
    main()
