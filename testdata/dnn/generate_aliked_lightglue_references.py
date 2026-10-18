#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
"""Generate the ALIKED/LightGlue NPY files introduced in opencv_extra#1366.

See generate_aliked_lightglue_references.md for dependencies and commands.
Inference uses ONNX Runtime CPU; OpenCV supplies the C++ test preprocessing.
Use --check with --test-binary to validate the C++ regressions as well.
"""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

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


def sort_features(keypoints, descriptors, scores):
    # ALIKED's top-k output order is not stable across inference engines, so
    # save a canonical order instead: keypoint position, x then y. Position is
    # the only stable key here; scores are not. Engines disagree on the scores
    # by up to 3e-5, which is more than the smallest gaps between neighbouring
    # scores (down to 0), while distinct keypoints are at least 0.4 px apart.
    order = np.lexsort((keypoints[:, 1], keypoints[:, 0]))
    return keypoints[order], descriptors[order], scores[order]


def extract_features(session, image, strict):
    blob = cv.dnn.blobFromImage(image, 1.0 / 255.0, (640, 640),
                               swapRB=True, crop=False, strictResize=strict)
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

    # Sort before the features reach either the NPY files or LightGlue, so the
    # saved match indices refer to this same canonical order.
    return sort_features(keypoints, descriptors, scores)


def matcher_keypoints(keypoints, image):
    # Reproduce ALIKED's [-1,1] -> original pixel coordinates and then
    # LightGlueMatcher's pixel -> [-1,1] conversion in OpenCV 5.x.
    # Feeding raw normalized keypoints directly skips float32 rounding.
    height, width = image.shape[:2]
    size = np.array([width, height], dtype=np.float32)
    pixels = (keypoints + np.float32(1.0)) * np.float32(0.5) * size
    return pixels / size * np.float32(2.0) - np.float32(1.0)


def generate(testdata, models_dir, resize):
    strict = (resize == "linear-exact")
    aliked = make_session(models_dir / "aliked-n16rot-top1k-640.onnx")
    references = {}
    features = []
    for name in ("box", "box_in_scene"):
        path = testdata / "cv" / "shared" / (name + ".png")
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Cannot read image: {}".format(path))
        keypoints, descriptors, scores = extract_features(aliked, image, strict)
        for label, values in (("keypoints", keypoints),
                              ("descriptors", descriptors), ("scores", scores)):
            references["aliked_{}_{}.npy".format(label, name)] = values
        features.append((matcher_keypoints(keypoints, image), descriptors))
    del aliked

    lightglue = make_session(models_dir / "aliked_lightglue.onnx")
    matches, mscores = lightglue.run(["matches0", "mscores0"], {
        "kpts0": features[0][0][None], "kpts1": features[1][0][None],
        "desc0": features[0][1][None], "desc1": features[1][1][None],
    })
    matches = matches.reshape(-1, 2)
    mscores = mscores.reshape(-1)

    # The match rows follow LightGlue's output order, so give them a canonical
    # order too. The features are position-sorted, so ordering by index pair is
    # ordering by the matched query keypoint and then the matched train one.
    order = np.lexsort((matches[:, 1], matches[:, 0]))
    references["lightglue_matches.npy"] = matches[order]
    references["lightglue_mscores.npy"] = mscores[order]
    return references


def check_cpp(test_binary, testdata, models_dir, output_dir):
    # The C++ tests resolve both images/models and references through one
    # testdata root. Stage links so --output-dir is tested without replacing
    # the checked-in files or accidentally reading references elsewhere.
    required = {"Features2d_ALIKED.Regression", "Features2d_LightGlue.Regression"}
    with tempfile.TemporaryDirectory(prefix="aliked-lightglue-check-") as directory:
        root = Path(directory)
        (root / "cv").symlink_to((testdata / "cv").resolve(), target_is_directory=True)
        dnn = root / "dnn"
        (dnn / "onnx").mkdir(parents=True)
        (dnn / "onnx" / "models").symlink_to(models_dir.resolve(), target_is_directory=True)
        for path in output_dir.glob("*.npy"):
            (dnn / path.name).symlink_to(path.resolve())
        report = root / "results.xml"
        env = os.environ.copy()
        env["OPENCV_TEST_DATA_PATH"] = str(root)
        print("C++ validation: {}; OPENCV_FORCE_DNN_ENGINE={}".format(
            test_binary, env.get("OPENCV_FORCE_DNN_ENGINE", "unset (ENGINE_OPENCV)")),
            flush=True)
        result = subprocess.run([
            str(test_binary.resolve()),
            "--gtest_filter=" + ":".join(sorted(required)),
            "--gtest_output=xml:" + str(report),
        ], env=env, check=False, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        print(result.stdout, end="", flush=True)
        # A zero exit status is insufficient: an unavailable/skipped test or
        # an empty filter must not silently count as C++ validation.
        # OpenCV's bundled Google Test records SkipTestException as a passing
        # XML testcase, but emits a SKIP line on stdout. Reject that as well.
        if (result.returncode != 0 or not report.is_file()
                or re.search(r"\[\s*SKIP\s*\]", result.stdout)):
            print("FAIL: C++ regression tests")
            return False
        completed = set()
        for case in ET.parse(report).iter("testcase"):
            name = case.get("classname", "") + "." + case.get("name", "")
            if (case.get("status") == "run" and case.get("result", "completed") == "completed"
                    and not case.get("custom_status")
                    and case.find("failure") is None and case.find("skipped") is None):
                completed.add(name)
        passed = required <= completed
        print("{}: C++ regression tests".format("PASS" if passed else "FAIL"))
        return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=Path(__file__).resolve().parents[1],
                        help="opencv_extra/testdata directory (default: relative to script)")
    parser.add_argument("--models-dir", type=Path,
                        help="model directory (default: TESTDATA/dnn/onnx/models)")
    parser.add_argument("--output-dir", type=Path,
                        help="NPY output/check directory (default: TESTDATA/dnn)")
    parser.add_argument("--resize", choices=("linear", "linear-exact"), default="linear",
                        help="linear for current 5.x; linear-exact only for opencv#30131")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true",
                        help="compare NPY files and run C++ regressions; requires --test-binary")
    checks.add_argument("--check-npy", action="store_true",
                        help="compare NPY files only; does not validate the C++ implementation")
    parser.add_argument("--test-binary", type=Path,
                        help="opencv_test_features executable for --check (engine from environment)")
    args = parser.parse_args()
    if args.check and (args.test_binary is None or not args.test_binary.is_file()):
        parser.error("--check requires --test-binary /path/to/opencv_test_features; "
                     "use --check-npy for Python/NPY comparison only")
    if args.test_binary is not None and not args.check:
        parser.error("--test-binary requires --check")
    checking = args.check or args.check_npy
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
    if not checking:
        output_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for name, values in references.items():
        path = output_dir / name
        if checking:
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
    if args.check and success:
        success = check_cpp(args.test_binary, args.testdata, models_dir, output_dir)
    elif args.check:
        print("C++ validation not run because NPY comparison failed.")
    elif args.check_npy:
        print("NPY comparison only; C++ regression tests were not run.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
