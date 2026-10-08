#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
"""Generate the DISK NPY files under testdata/cv/features/disk.

See generate_disk_references.md for dependencies and commands.
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


def canonical_order(xy):
    # DISK's top-N is picked with an unstable partial_sort, so neither side has
    # a reproducible row order; test_disk.cpp compares both in position order.
    return np.lexsort((xy[:, 1], xy[:, 0]))


def discover_tags(reference_dir):
    # Tags are whatever references are actually committed, not a fixed list.
    tags = [m.group(1) for m in (re.fullmatch(r"box_in_scene_(.+)_kpts\.npy", path.name)
                                 for path in sorted(reference_dir.glob("box_in_scene_*_kpts.npy")))
            if m]
    if not tags:
        raise FileNotFoundError("No box_in_scene_*_kpts.npy references under {}".format(reference_dir))
    return tags


def net_size_for_tag(tag, opencv_src):
    # Most tags spell out cv::Size(width, height) in their own name.
    m = re.fullmatch(r"(\d+)x(\d+)", tag)
    if m:
        return int(m.group(1)), int(m.group(2))
    # "default" has no size in its name: read it from the C++ source instead.
    disk_cpp = opencv_src / "modules" / "features" / "src" / "disk.cpp"
    m = re.search(r"kDefaultDiskInputSize\s*=\s*Size\(\s*(\d+)\s*,\s*(\d+)\s*\)", disk_cpp.read_text())
    if not m:
        raise ValueError("Could not find kDefaultDiskInputSize in {}".format(disk_cpp))
    return int(m.group(1)), int(m.group(2))


def load_existing_counts(reference_dir, tags):
    # test_disk.cpp reads n = refKpts.rows from the committed file for each tag.
    return {tag: np.load(reference_dir / "box_in_scene_{}_kpts.npy".format(tag),
                        allow_pickle=False).shape[0]
            for tag in tags}


def make_session(path):
    # Match OpenCV's CPU ORT engine, which uses default session options.
    options = ort.SessionOptions()
    print("Model: {} (SHA-1 {})".format(
        path, hashlib.sha1(path.read_bytes()).hexdigest()))
    return ort.InferenceSession(str(path), options,
                               providers=["CPUExecutionProvider"])


def extract_features(session, image, net_size, interpolation):
    # Current 5.x DISK calls blobFromImage directly (INTER_LINEAR internally).
    # For the strictResize variant, resize uint8 BGR before conversion to float:
    # INTER_LINEAR_EXACT falls back to INTER_LINEAR for float images.
    resized = image if interpolation == cv.INTER_LINEAR else cv.resize(
        image, net_size, interpolation=interpolation)
    blob = cv.dnn.blobFromImage(resized, 1.0 / 255.0, net_size,
                               swapRB=True, crop=False)
    keypoints, scores, descriptors = session.run(
        ["keypoints", "scores", "descriptors"], {"image": blob})
    keypoints = keypoints.reshape(-1, 2).astype(np.float32)
    scores = scores.reshape(-1).astype(np.float32)
    descriptors = descriptors.reshape(-1, 128).astype(np.float32)
    if not (len(keypoints) == len(scores) == len(descriptors)):
        raise ValueError("DISK outputs have inconsistent feature counts")
    return keypoints, scores, descriptors


def generate(testdata, models_dir, opencv_src, resize):
    # Tags, their net sizes and their keypoint counts all come from the
    # checked-in references (plus disk.cpp for the "default" net size), not
    # from --output-dir: a scratch output dir has nothing to derive them from.
    interpolation = cv.INTER_LINEAR if resize == "linear" else cv.INTER_LINEAR_EXACT
    reference_dir = testdata / "cv" / "features" / "disk"
    tags = discover_tags(reference_dir)
    counts = load_existing_counts(reference_dir, tags)
    net_sizes = {tag: net_size_for_tag(tag, opencv_src) for tag in tags}
    session = make_session(models_dir / "disk.onnx")
    path = testdata / "cv" / "shared" / "box_in_scene.png"
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Cannot read image: {}".format(path))
    height, width = image.shape[:2]

    references = {}
    for tag, net_size in net_sizes.items():
        keypoints, scores, descriptors = extract_features(
            session, image, net_size, interpolation)

        # DISK_Impl::detectAndCompute: keep score > scoreThreshold_ (0.0f in
        # the regression test), then rescale net-space pixel coordinates back
        # to the original image size.
        keep = scores > 0.0
        keypoints = keypoints[keep]
        scores = scores[keep]
        descriptors = descriptors[keep]
        net_w, net_h = net_size
        scale = np.array([width / net_w, height / net_h], dtype=np.float32)
        keypoints = keypoints * scale

        # setMaxKeypoints(n): partial_sort descending by response, keep top n.
        n = counts[tag]
        if len(scores) < n:
            raise ValueError("Only {} keypoints available for tag {}, need {}".format(
                len(scores), tag, n))
        order = np.argsort(-scores, kind="stable")[:n]
        references["box_in_scene_{}_kpts.npy".format(tag)] = np.concatenate(
            [keypoints[order], scores[order, None]], axis=1).astype(np.float32)
        references["box_in_scene_{}_desc.npy".format(tag)] = descriptors[order]
    del session
    return references


def check_cpp(test_binary, testdata, models_dir, output_dir, tags):
    # The C++ tests resolve images/models through "cv/..." and "dnn/..." roots
    # and references through "features/disk/..." (also under "cv/"). Stage
    # links so --output-dir is tested without replacing the checked-in files
    # or accidentally reading references elsewhere.
    required = {"Features2d_DISK.regression_{}".format(tag) for tag in tags}
    with tempfile.TemporaryDirectory(prefix="disk-check-") as directory:
        root = Path(directory)
        cv_dir = root / "cv"
        cv_dir.mkdir(parents=True)
        (cv_dir / "shared").symlink_to(
            (testdata / "cv" / "shared").resolve(), target_is_directory=True)
        (cv_dir / "features").mkdir()
        (cv_dir / "features" / "disk").symlink_to(
            output_dir.resolve(), target_is_directory=True)
        dnn = root / "dnn"
        dnn.mkdir()
        (dnn / "disk.onnx").symlink_to((models_dir / "disk.onnx").resolve())
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
    parser.add_argument("--opencv-src", type=Path,
                        help="opencv checkout, for reading disk.cpp's default net size "
                             "(default: TESTDATA/../../opencv, a sibling checkout)")
    parser.add_argument("--models-dir", type=Path,
                        help="model directory (default: TESTDATA/dnn)")
    parser.add_argument("--output-dir", type=Path,
                        help="NPY output/check directory (default: TESTDATA/cv/features/disk)")
    parser.add_argument("--resize", choices=("linear", "linear-exact"), default="linear",
                        help="resize interpolation: current 5.x (linear) or a build with "
                             "strictResize enabled in DISK (linear-exact)")
    parser.add_argument("--check", action="store_true",
                        help="compare against committed references and run the C++ regressions")
    parser.add_argument("--check-npy", action="store_true",
                        help="compare against committed references only (no C++ run)")
    parser.add_argument("--test-binary", type=Path,
                        help="path to opencv_test_features (required with --check)")
    args = parser.parse_args()
    args.testdata = args.testdata.resolve()

    if args.check and (args.test_binary is None or not args.test_binary.is_file()):
        parser.error("--check requires --test-binary /path/to/opencv_test_features; "
                     "use --check-npy for Python/NPY comparison only")
    if args.test_binary is not None and not args.check:
        parser.error("--test-binary requires --check")
    checking = args.check or args.check_npy
    models_dir = args.models_dir or args.testdata / "dnn"
    output_dir = args.output_dir or args.testdata / "cv" / "features" / "disk"
    opencv_src = args.opencv_src or args.testdata.parent.parent / "opencv"
    print("OpenCV {}; NumPy {}; ONNX Runtime {}; resize={}".format(
        cv.__version__, np.__version__, ort.__version__, args.resize))
    references = generate(args.testdata, models_dir, opencv_src, args.resize)

    # Compute everything before writing any files, so inference errors do not
    # leave a mixture of old and new references.
    for name, values in references.items():
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Invalid dtype or non-finite values in {}".format(name))
    if not checking:
        output_dir.mkdir(parents=True, exist_ok=True)

    success = True
    if checking:
        # Compare kpts+desc together, in canonical position order, matching
        # how test_disk.cpp itself compares (row order is not reproducible).
        tags = discover_tags(args.testdata / "cv" / "features" / "disk")
        for tag in tags:
            kpts_name = "box_in_scene_{}_kpts.npy".format(tag)
            desc_name = "box_in_scene_{}_desc.npy".format(tag)
            kpts_path, desc_path = output_dir / kpts_name, output_dir / desc_name
            if not kpts_path.is_file() or not desc_path.is_file():
                print("MISSING: {} or {}".format(kpts_path, desc_path))
                success = False
                continue
            new_kpts, new_desc = references[kpts_name], references[desc_name]
            ref_kpts, ref_desc = (np.load(kpts_path, allow_pickle=False),
                                  np.load(desc_path, allow_pickle=False))
            compatible = (ref_kpts.shape == new_kpts.shape and ref_kpts.dtype == new_kpts.dtype
                          and ref_desc.shape == new_desc.shape and ref_desc.dtype == new_desc.dtype)
            if not compatible:
                print("FAIL: {} shape/dtype mismatch".format(tag))
                success = False
                continue
            new_order, ref_order = canonical_order(new_kpts[:, :2]), canonical_order(ref_kpts[:, :2])
            new_kpts, new_desc = new_kpts[new_order], new_desc[new_order]
            ref_kpts, ref_desc = ref_kpts[ref_order], ref_desc[ref_order]

            kpts_tol, desc_tol = 1e-4, 1e-5
            kpts_passed = np.allclose(new_kpts, ref_kpts, rtol=0, atol=kpts_tol)
            desc_passed = np.allclose(new_desc, ref_desc, rtol=0, atol=desc_tol)
            kpts_diff = np.max(np.abs(new_kpts - ref_kpts)) if new_kpts.size else 0.0
            desc_diff = np.max(np.abs(new_desc - ref_desc)) if new_desc.size else 0.0
            print("{}: {} (max abs diff={:.9g}, atol={})".format(
                "PASS" if kpts_passed else "FAIL", kpts_name, kpts_diff, kpts_tol))
            print("{}: {} (max abs diff={:.9g}, atol={})".format(
                "PASS" if desc_passed else "FAIL", desc_name, desc_diff, desc_tol))
            success = success and kpts_passed and desc_passed
    else:
        for name, values in references.items():
            path = output_dir / name
            np.save(path, np.ascontiguousarray(values), allow_pickle=False)
            print("Saved {}: {} {}".format(path, values.shape, values.dtype))
    if args.check and success:
        success = check_cpp(args.test_binary, args.testdata, models_dir, output_dir, tags)
    elif args.check:
        print("C++ validation not run because NPY comparison failed.")
    elif args.check_npy:
        print("NPY comparison only; C++ regression tests were not run.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
