# ALIKED/LightGlue references

Generate the eight NPY files from [#1366](https://github.com/opencv/opencv_extra/pull/1366)
using ONNX Runtime CPU and OpenCV preprocessing. Requires `numpy`,
`opencv-python` (or an OpenCV Python build), and `onnxruntime`. Match the ONNX
Runtime version to the C++ build; Python 3.11+ is needed for ORT 1.25.1.

## Resize and engine selection

Current OpenCV `5.x` ALIKED uses `blobFromImage` with `INTER_LINEAR`. Use the
script's default `--resize linear` for that branch. `--resize linear-exact`
changes the input pixels and is **only** for a build with
[opencv#30131](https://github.com/opencv/opencv/pull/30131), which enables
`strictResize` in ALIKED. The OpenCV version string does not identify which
preprocessing a build uses. Do not use exact resize with an unpatched `5.x` build.

Inference in this script uses ORT CPU. The C++ tests default to `ENGINE_OPENCV`,
which can produce different feature ordering and numerical results. To validate
ORT references with the C++ ORT engine, build OpenCV with `WITH_ONNXRUNTIME=ON`
and set `OPENCV_FORCE_DNN_ENGINE=2`. The script never changes the test engine;
leave the variable unset to check the default OpenCV engine.

## Commands

From `testdata/dnn`:

```sh
python download_models.py aliked

# Regenerate for current 5.x, without replacing the checked-in references.
python generate_aliked_lightglue_references.py --output-dir /tmp/aliked-linear

# Check all eight files AND run both C++ regressions against those files.
OPENCV_FORCE_DNN_ENGINE=2 python generate_aliked_lightglue_references.py \
    --output-dir /tmp/aliked-linear --check \
    --test-binary /path/to/opencv-build/bin/opencv_test_features

# Reproduce references for a build containing opencv#30131.
python generate_aliked_lightglue_references.py \
    --resize linear-exact --output-dir /tmp/aliked-exact
OPENCV_FORCE_DNN_ENGINE=2 python generate_aliked_lightglue_references.py \
    --resize linear-exact --output-dir /tmp/aliked-exact --check \
    --test-binary /path/to/strict-resize-build/bin/opencv_test_features

# Python/NPY comparison only; this cannot establish C++ compatibility.
python generate_aliked_lightglue_references.py --check-npy
```

Inputs are the downloaded models and `testdata/cv/shared/{box,box_in_scene}.png`.
Use `--testdata` and `--models-dir` to override input locations. Omit `--output-dir`
to replace the eight files in `testdata/dnn`. Regenerate all eight together.

`--check` requires `--test-binary`, writes no reference files, and exits 1 on
NPY mismatch or C++ failure. It stages a temporary testdata tree so both C++ tests
read the selected images, models, and `--output-dir` references. Both
`Features2d_ALIKED.Regression` and `Features2d_LightGlue.Regression` must actually
run and pass; skipped or missing tests fail validation. An exact-resize NPY set
can reproduce perfectly in Python and still fail a C++ build using linear resize.

`--check-npy` compares Python-generated arrays only. Indices must match exactly;
float tolerances are `1e-5`, or `1e-4` for LightGlue confidence scores. It prints
explicitly that C++ tests were not run. ORT versions can change scores within
small numerical differences even when keypoints, descriptors, and matches agree.

## Validation

Tested with Python 3.13, OpenCV Python 4.13.0, NumPy 2.4.5, and ORT 1.25.1:

| C++ build | References | Engine | C++ result |
| --- | --- | --- | --- |
| `5.x` at `c90bebd67b` | Linear (211 matches) | ORT / OpenCV | Both pass |
| `5.x` at `c90bebd67b` | Linear exact (209 matches) | ORT | Both fail; `--check` exits 1 |
| #30131 at `dc42d35511` | Linear exact (209 matches) | ORT | Both pass |

The original eight committed references pass `--check-npy` with OpenCV
5.0.0-pre, NumPy 1.26.4, and ORT 1.20.0. Validation also rejects skipped tests
and a filter that runs no tests, even when the executable exits 0.
