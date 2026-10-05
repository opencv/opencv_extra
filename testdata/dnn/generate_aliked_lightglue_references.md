# ALIKED/LightGlue references

Generate the eight NPY files from [#1366](https://github.com/opencv/opencv_extra/pull/1366)
using ONNX Runtime CPU and OpenCV preprocessing.

Requires `numpy`, `opencv-python` (or an OpenCV Python build), and `onnxruntime`.
Tested environments:

| Mode | Python | OpenCV | NumPy | ONNX Runtime |
| --- | --- | --- | --- | --- |
| Linear | 3.10 | 5.0.0-pre | 1.26.4 | 1.20.0 |
| Linear exact | 3.13 | 4.13.0 | 2.4.5 | 1.25.1 |

Match the ONNX Runtime version to the C++ build. Version 1.25.1 requires Python 3.11+.

From `testdata/dnn`:

```sh
python download_models.py aliked

# Check the original references.
python generate_aliked_lightglue_references.py --resize linear --check

# Generate references for opencv/opencv#30131.
python generate_aliked_lightglue_references.py --resize linear-exact --output-dir /tmp/aliked-exact
```

Inputs are the downloaded models and `testdata/cv/shared/{box,box_in_scene}.png`.
Use `--testdata` and `--models-dir` to override input locations. Omit `--output-dir`
to replace the files in `testdata/dnn`; regenerate all eight together using the
interpolation selected by your OpenCV branch.

The script resizes uint8 images before `blobFromImage`, normalizes descriptors,
and reproduces the C++ coordinate conversions. It prints versions and model
hashes. `--check` writes nothing and exits 1 on mismatch: indices must match
exactly; float tolerances are `1e-5`, or `1e-4` for LightGlue confidence scores.

## Validation

All eight original references pass comparison (211 matches). Exact linear
interpolation produces 209 matches and identical arrays on repeated runs.
Both C++ tests pass with `ENGINE_ORT` on opencv/opencv#30131 at `dc42d35511`
(OpenCV 5.1.0-dev, ONNX Runtime 1.25.1):

```sh
OPENCV_FORCE_DNN_ENGINE=2 OPENCV_TEST_DATA_PATH=/path/to/updated/testdata \
    /path/to/opencv-build/bin/opencv_test_features \
    --gtest_filter='Features2d_ALIKED.Regression:Features2d_LightGlue.Regression'
```

This requires `WITH_ONNXRUNTIME=ON`. The default `ENGINE_OPENCV` fails both
ordered comparisons: it swaps rows 661/662 and 953/954 for `box`, and 630/631
for `box_in_scene`, despite identical input blobs. Exact resize does not resolve
this cross-engine ordering difference.
