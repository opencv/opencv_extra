#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
"""Generate the ALIKED/LightGlue NPY files introduced in opencv_extra#1366.

See generate_aliked_lightglue_references.md for dependencies and commands.
Inference uses ONNX Runtime CPU; OpenCV supplies the C++ test preprocessing.
"""

import argparse
import hashlib
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort


def make_session(path):
    # Match OpenCV's CPU ORT engine, which uses default session options.
    options = ort.SessionOptions()
    print("Model: {} (SHA-1 {})".format(
        path, hashlib.sha1(path.read_bytes()).hexdigest()))
    return ort.InferenceSession(str(path), options,
                               providers=["CPUExecutionProvider"])


def extract_features(session, image, interpolation):
    # Resize uint8 BGR BEFORE conversion to float. INTER_LINEAR_EXACT falls
    # back to INTER_LINEAR for float images. crop=False stretches to 640x640.
    resized = cv.resize(image, (640, 640), interpolation=interpolation)
    blob = cv.dnn.blobFromImage(resized, 1.0 / 255.0, (640, 640),
                               swapRB=True, crop=False)
    keypoints, descriptors, scores = session.run(
        ["keypoints", "descriptors", "scores"], {"image": blob})
    keypoints = keypoints.reshape(-1, 2)
    descriptors = descriptors.reshape(-1, 128)
    scores = scores.reshape(-1)
    if not (len(keypoints) == len(descriptors) == len(scores)):
        raise ValueError("ALIKED outputs have inconsistent feature counts")

    # ALIKED::Params::normalizeDescriptors defaults to true. Use OpenCV's
    # normalization, including its double-precision norm accumulation.
    descriptors = descriptors.copy()
    for row in descriptors:
        row[:] = cv.normalize(row, None).reshape(-1)
    return keypoints, descriptors, scores


def matcher_keypoints(keypoints, image):
    # Reproduce ALIKED's [-1,1] -> original pixel coordinates and then
    # LightGlueMatcher's pixel -> [-1,1] conversion in OpenCV 5.x.
    # Feeding raw normalized keypoints directly skips float32 rounding.
    height, width = image.shape[:2]
    size = np.array([width, height], dtype=np.float32)
    pixels = (keypoints + np.float32(1.0)) * np.float32(0.5) * size
    return pixels / size * np.float32(2.0) - np.float32(1.0)


def generate(testdata, models_dir, resize):
    interpolation = cv.INTER_LINEAR if resize == "linear" else cv.INTER_LINEAR_EXACT
    aliked = make_session(models_dir / "aliked-n16rot-top1k-640.onnx")
    references = {}
    features = []
    for name in ("box", "box_in_scene"):
        path = testdata / "cv" / "shared" / (name + ".png")
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Cannot read image: {}".format(path))
        keypoints, descriptors, scores = extract_features(aliked, image, interpolation)
        for label, values in (("keypoints", keypoints),
                              ("descriptors", descriptors), ("scores", scores)):
            references["aliked_{}_{}.npy".format(label, name)] = values
        features.append((matcher_keypoints(keypoints, image), descriptors))
    del aliked

    lightglue = make_session(models_dir / "aliked_lightglue.onnx")
    matches, scores = lightglue.run(["matches0", "mscores0"], {
        "kpts0": features[0][0][None], "kpts1": features[1][0][None],
        "desc0": features[0][1][None], "desc1": features[1][1][None],
    })
    references["lightglue_matches.npy"] = matches.reshape(-1, 2)
    references["lightglue_mscores.npy"] = scores.reshape(-1)
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=Path(__file__).resolve().parents[1],
                        help="opencv_extra/testdata directory (default: relative to script)")
    parser.add_argument("--models-dir", type=Path,
                        help="model directory (default: TESTDATA/dnn/onnx/models)")
    parser.add_argument("--output-dir", type=Path,
                        help="NPY output/check directory (default: TESTDATA/dnn)")
    parser.add_argument("--resize", choices=("linear", "linear-exact"), default="linear",
                        help="linear for the original references; linear-exact for opencv#30131")
    parser.add_argument("--check", action="store_true",
                        help="compare existing NPY files without writing; exit 1 on mismatch")
    args = parser.parse_args()
    models_dir = args.models_dir or args.testdata / "dnn" / "onnx" / "models"
    output_dir = args.output_dir or args.testdata / "dnn"
    print("OpenCV {}; NumPy {}; ONNX Runtime {}; resize={}".format(
        cv.__version__, np.__version__, ort.__version__, args.resize))
    references = generate(args.testdata, models_dir, args.resize)

    # Compute everything before writing any files, so inference errors do not
    # leave a mixture of new ALIKED and old LightGlue references.
    for name, values in references.items():
        expected_dtype = np.int64 if name == "lightglue_matches.npy" else np.float32
        if values.dtype != expected_dtype or not np.isfinite(values).all():
            raise ValueError("Invalid dtype or non-finite values in {}".format(name))
    if not args.check:
        output_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for name, values in references.items():
        path = output_dir / name
        if args.check:
            if not path.is_file():
                print("MISSING: {}".format(path))
                success = False
                continue
            reference = np.load(path, allow_pickle=False)
            compatible = reference.shape == values.shape and reference.dtype == values.dtype
            if not compatible:
                print("FAIL: {} shape/dtype: {} {} vs {} {}".format(
                    name, values.shape, values.dtype, reference.shape, reference.dtype))
                success = False
                continue
            if values.dtype == np.int64:
                passed = np.array_equal(values, reference)
                detail = "exact index comparison"
            else:
                tolerance = 1e-4 if name == "lightglue_mscores.npy" else 1e-5
                passed = np.allclose(values, reference, rtol=0, atol=tolerance)
                difference = np.max(np.abs(values - reference)) if values.size else 0.0
                detail = "max abs diff={:.9g}, atol={}".format(difference, tolerance)
            print("{}: {} ({})".format("PASS" if passed else "FAIL", name, detail))
            success = success and passed
        else:
            np.save(path, np.ascontiguousarray(values), allow_pickle=False)
            print("Saved {}: {} {}".format(path, values.shape, values.dtype))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
