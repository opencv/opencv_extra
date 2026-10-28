#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
"""Generate features/disk/disk_lightglue_matches.npy, the DISK/LightGlue reference.

See generate_disk_lightglue_references.md for dependencies and commands.
Inference uses OpenCV DNN from the imported cv2, the engine the C++ test runs,
so use the Python bindings of the OpenCV build under test. Unlike
generate_aliked_lightglue_references.py this script does not use ONNX Runtime:
the match indices point into DISK's own response-ordered keypoint list, and
generate_disk_references.md measures ORT's DISK scores drifting far enough from
ENGINE_OPENCV's to reshuffle that list.
Use --check with --test-binary to validate the C++ regression as well.
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

# Mirrors Features2d_LightGlue_DISK.Regression: DISK::create(diskPath) keeps the
# default scoreThreshold of 0.0f, and the matcher is created with a 0.0f match
# confidence threshold. DISK_Impl drops score <= threshold, the matcher keeps
# score >= threshold.
DISK_SCORE_THRESHOLD = 0.0
LG_SCORE_THRESHOLD = 0.0

# The same test's tolerances, applied by --check/--check-npy so a Python
# comparison answers the same question the C++ regression does.
COUNT_TOLERANCE = 2
MISMATCH_TOLERANCE = 3

# The same test calls disk->setMaxKeypoints(1024) and then asserts that both
# images yield exactly that many keypoints, so the saved indices run over the
# 1024 strongest detections per image.
TEST_MAX_KEYPOINTS = 1024


def make_net(path):
    # Load the models the way DISK and LightGlueMatcher do, with
    # readNetFromONNX and default backend/target.
    if not path.is_file():
        raise FileNotFoundError("Cannot read model: {} (run download_models.py disk)".format(path))
    print("Model: {} (SHA-1 {})".format(
        path, hashlib.sha1(path.read_bytes()).hexdigest()))
    return cv.dnn.readNetFromONNX(str(path))


def sort_features(keypoints, descriptors, scores, max_keypoints):
    # DISK_Impl::detectAndCompute's setMaxKeypoints(n) step: partial_sort the
    # detections by response, strongest first, and keep the first n. The match
    # indices saved below point into exactly this order, so unlike the ALIKED
    # script the features must NOT be reordered by position before matching.
    # std::partial_sort is unstable, so features sharing a score can come out
    # in a different order here than in C++; that is why the C++ test compares
    # match sets with a tolerance instead of row by row.
    order = np.argsort(-scores, kind="stable")
    if max_keypoints > 0:
        if len(order) < max_keypoints:
            print("WARNING: only {} keypoints available, but the test asserts {}; "
                  "the reference will not match it".format(len(order), max_keypoints))
        order = order[:max_keypoints]
    return keypoints[order], descriptors[order], scores[order]


def extract_features(net, image, net_size, interpolation, max_keypoints):
    blob = cv.dnn.blobFromImage(image, 1.0 / 255.0, net_size,
                               swapRB=True, crop=False, interpolation=interpolation)
    net.setInput(blob, "image")
    keypoints, scores, descriptors = net.forward(["keypoints", "scores", "descriptors"])
    keypoints = keypoints.reshape(-1, 2)
    scores = scores.reshape(-1).astype(np.float32)
    descriptors = descriptors.reshape(-1, 128).astype(np.float32)
    if not (len(keypoints) == len(scores) == len(descriptors)):
        raise ValueError("DISK outputs have inconsistent feature counts")

    # DISK_Impl::detectAndCompute: keep score > scoreThreshold_, then rescale the
    # net-space integer pixel coordinates back to the original image size.
    keep = scores > DISK_SCORE_THRESHOLD
    detections = len(scores)
    keypoints = keypoints[keep].astype(np.float32)
    scores = scores[keep]
    descriptors = descriptors[keep]
    if len(scores) == 0:
        raise ValueError("DISK found no keypoints above the score threshold")

    net_w, net_h = net_size
    height, width = image.shape[:2]
    scale = np.array([np.float32(width) / np.float32(net_w),
                      np.float32(height) / np.float32(net_h)], dtype=np.float32)
    keypoints = keypoints * scale

    keypoints, descriptors, scores = sort_features(
        keypoints, descriptors, scores, max_keypoints)
    print("{} detections, {} above the score threshold, {} kept for matching"
          " (strongest {:.6g}, weakest kept {:.6g})".format(
              detections, int(np.count_nonzero(keep)), len(scores),
              scores[0], scores[-1]))
    return keypoints, descriptors


def matcher_keypoints(keypoints, image):
    # normalizeLightGlueKeypoints() for LG_DISK: pixel coordinates to [0, 1]
    # over size - 1, in float32, with no offset.
    height, width = image.shape[:2]
    if width < 2 or height < 2:
        raise ValueError("DISK LightGlue normalization needs an image at least 2x2")
    scale = np.array([np.float32(1.0) / np.float32(width - 1),
                      np.float32(1.0) / np.float32(height - 1)], dtype=np.float32)
    return keypoints * scale


def generate(testdata, models_dir, opencv_src, resize, max_keypoints):
    interpolation = cv.INTER_LINEAR if resize == "linear" else cv.INTER_LINEAR_EXACT
    net_size = (1024, 1024)
    print("DISK net size={}x{}; maxKeypoints={}".format(
        net_size[0], net_size[1], max_keypoints))
    disk = make_net(models_dir / "disk.onnx")
    features = []
    for name in ("box", "box_in_scene"):
        path = testdata / "cv" / "shared" / (name + ".png")
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Cannot read image: {}".format(path))
        keypoints, descriptors = extract_features(
            disk, image, net_size, interpolation, max_keypoints)
        features.append((matcher_keypoints(keypoints, image), descriptors))
    del disk

    # match(descs_box, descs_box_in_scene): box is the query side, box_in_scene
    # the train side, so kpts0/desc0 come from box.
    lightglue = make_net(models_dir / "onnx" / "models" / "disk_lightglue.onnx")
    (kpts0, desc0), (kpts1, desc1) = features
    lightglue.setInput(kpts0[None], "kpts0")
    lightglue.setInput(kpts1[None], "kpts1")
    lightglue.setInput(desc0[None], "desc0")
    lightglue.setInput(desc1[None], "desc1")
    outs = lightglue.forward(["matches0", "matches1", "mscores0", "mscores1"])
    del lightglue

    # DISKLightGlueMatcherImpl reads matches0/mscores0 only: for each query
    # keypoint i, the matched train keypoint j (or -1) and its confidence.
    query_count, train_count = len(kpts0), len(kpts1)
    matches0 = outs[0].reshape(-1).astype(np.int64)
    mscores0 = outs[2].reshape(-1).astype(np.float32)
    if not (len(matches0) == len(mscores0) == query_count):
        raise ValueError("LightGlue outputs do not cover all {} query keypoints".format(
            query_count))
    valid = ((matches0 >= 0) & (matches0 < train_count)
             & (mscores0 > LG_SCORE_THRESHOLD))
    matches = np.stack([np.arange(query_count, dtype=np.int64)[valid],
                        matches0[valid]], axis=1)
    print("matches:", matches.shape)

    # The matcher emits one match per query keypoint in ascending query order;
    # sort by index pair anyway so the saved rows are canonical regardless.
    order = np.lexsort((matches[:, 1], matches[:, 0]))
    return {"disk_lightglue_matches.npy": matches[order]}


def compare_matches(values, reference):
    # Features2d_LightGlue_DISK.Regression compares match *sets* with a
    # tolerance, because DNN engine differences reorder equivalent matches.
    new_set = {tuple(row) for row in values.tolist()}
    ref_set = {tuple(row) for row in reference.tolist()}
    ref_not_found = len(ref_set - new_set)
    act_not_found = len(new_set - ref_set)
    count_diff = abs(len(values) - len(reference))
    passed = (count_diff <= COUNT_TOLERANCE
              and ref_not_found <= MISMATCH_TOLERANCE
              and act_not_found <= MISMATCH_TOLERANCE)
    detail = ("{} vs {} matches (count diff={} <= {}), {} reference missing, "
              "{} extra (<= {} each)".format(
                  len(values), len(reference), count_diff, COUNT_TOLERANCE,
                  ref_not_found, act_not_found, MISMATCH_TOLERANCE))
    if passed and not ref_not_found and not act_not_found and count_diff == 0:
        detail += "; identical match set"
    return passed, detail


def check_cpp(test_binary, testdata, models_dir, output_dir):
    # The C++ test resolves the image through "shared/..." and the reference
    # through "features/disk/..." (both under "cv/"), and the models through
    # "dnn/...". Stage links so --output-dir is tested without replacing the
    # checked-in file or accidentally reading a reference elsewhere.
    required = {"Features2d_LightGlue_DISK.Regression"}
    with tempfile.TemporaryDirectory(prefix="disk-lightglue-check-") as directory:
        root = Path(directory)
        cv_dir = root / "cv"
        cv_dir.mkdir(parents=True)
        (cv_dir / "shared").symlink_to(
            (testdata / "cv" / "shared").resolve(), target_is_directory=True)
        (cv_dir / "features").mkdir()
        (cv_dir / "features" / "disk").symlink_to(
            output_dir.resolve(), target_is_directory=True)
        dnn = root / "dnn"
        (dnn / "onnx" / "models").mkdir(parents=True)
        (dnn / "disk.onnx").symlink_to((models_dir / "disk.onnx").resolve())
        (dnn / "onnx" / "models" / "disk_lightglue.onnx").symlink_to(
            (models_dir / "onnx" / "models" / "disk_lightglue.onnx").resolve())
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
            print("FAIL: C++ regression test")
            return False
        completed = set()
        for case in ET.parse(report).iter("testcase"):
            name = case.get("classname", "") + "." + case.get("name", "")
            if (case.get("status") == "run" and case.get("result", "completed") == "completed"
                    and not case.get("custom_status")
                    and case.find("failure") is None and case.find("skipped") is None):
                completed.add(name)
        passed = required <= completed
        print("{}: C++ regression test".format("PASS" if passed else "FAIL"))
        return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testdata", type=Path, default=Path(__file__).resolve().parents[1],
                        help="opencv_extra/testdata directory (default: relative to script)")
    parser.add_argument("--opencv-src", type=Path,
                        help="opencv checkout, for reading DISK's default net size, its resize "
                             "and the test's keypoint limit "
                             "(default: TESTDATA/../../opencv, a sibling checkout)")
    parser.add_argument("--models-dir", type=Path,
                        help="model directory holding disk.onnx and onnx/models/"
                             "disk_lightglue.onnx (default: TESTDATA/dnn)")
    parser.add_argument("--output-dir", type=Path,
                        help="NPY output/check directory (default: TESTDATA/cv/features/disk)")
    parser.add_argument("--max-keypoints", type=int, default=TEST_MAX_KEYPOINTS,
                        help="strongest keypoints per image kept for matching; 0 keeps every "
                             "detection (default: %(default)s, the test's setMaxKeypoints value)")
    parser.add_argument("--resize", choices=("linear", "linear-exact"), default="linear-exact",
                        help="resize interpolation used for DISK preprocessing")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true",
                        help="compare the NPY file and run the C++ regression; requires --test-binary")
    checks.add_argument("--check-npy", action="store_true",
                        help="compare the NPY file only; does not validate the C++ implementation")
    parser.add_argument("--test-binary", type=Path,
                        help="opencv_test_features executable for --check (engine from environment)")
    args = parser.parse_args()
    args.testdata = args.testdata.resolve()
    if args.check and (args.test_binary is None or not args.test_binary.is_file()):
        parser.error("--check requires --test-binary /path/to/opencv_test_features; "
                     "use --check-npy for Python/NPY comparison only")
    if args.test_binary is not None and not args.check:
        parser.error("--test-binary requires --check")
    if args.max_keypoints < 0:
        parser.error("--max-keypoints must be positive, or 0 to keep every detection")
    checking = args.check or args.check_npy
    models_dir = args.models_dir or args.testdata / "dnn"
    output_dir = args.output_dir or args.testdata / "cv" / "features" / "disk"
    opencv_src = args.opencv_src or args.testdata.parent.parent / "opencv"
    print("OpenCV {} ({}); NumPy {}; resize={}; maxKeypoints={}".format(
        cv.__version__, cv.__file__, np.__version__, args.resize,
        args.max_keypoints or "all"))
    references = generate(args.testdata, models_dir, opencv_src, args.resize, args.max_keypoints)

    # Compute everything before writing any file, so an inference error cannot
    # leave a truncated or half-written reference behind.
    limit = args.max_keypoints if args.max_keypoints > 0 else None
    for name, values in references.items():
        if (values.dtype != np.int64 or values.ndim != 2 or values.shape[1] != 2
                or values.size == 0 or values.min() < 0
                or (limit is not None and values.max() >= limit)):
            raise ValueError("Invalid shape, dtype or index range in {}".format(name))
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
            if reference.dtype != values.dtype or reference.ndim != 2 or reference.shape[1] != 2:
                print("FAIL: {} shape/dtype: {} {} vs {} {}".format(
                    name, values.shape, values.dtype, reference.shape, reference.dtype))
                success = False
                continue
            passed, detail = compare_matches(values, reference)
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
        print("NPY comparison only; C++ regression test was not run.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
