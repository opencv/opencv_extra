# ALIKED and LightGlue reference generation

`generate_aliked_lightglue_references.py` reconstructs the eight NPY references
added by [opencv_extra#1366](https://github.com/opencv/opencv_extra/pull/1366).
They are consumed by `modules/features/test/test_aliked_lightglue.cpp` in OpenCV.
The original generator was not included in that PR; this script reconstructs
the pipeline from the ONNX models and the OpenCV implementation.

## Dependencies and inputs

Install Python packages `numpy`, `opencv-python`, and `onnxruntime`, or use an
existing OpenCV Python build. The script does not require the new `strictResize`
Python argument or the ALIKED/LightGlue Python bindings. These environments were
validated on x86-64 Linux:

| Reference mode | Python | OpenCV | NumPy | ONNX Runtime |
| --- | --- | --- | --- | --- |
| Original, ordinary linear | 3.10 | 5.0.0-pre | 1.26.4 | 1.20.0 |
| Exact linear for opencv#30131 | 3.13 | 4.13.0 | 2.4.5 | 1.25.1 |

ONNX Runtime uses the CPU execution provider and default session options,
matching OpenCV's ORT engine. Use the same ONNX Runtime version as the C++ build
when validating regenerated references; 1.25.1 requires Python 3.11 or later.
The script prints package versions and model SHA-1 hashes for reproducibility.

From `testdata/dnn`, download the same models as the regression tests:

```sh
python download_models.py aliked
```

The files must be under `testdata/dnn/onnx/models`:

| Model | SHA-1 from download_models.py |
| --- | --- |
| aliked-n16rot-top1k-640.onnx | 41faa7bf5d7eb68a2851471ba03aa20c9db30e4c |
| aliked_lightglue.onnx | 02723aa521990e57fe33d90b67977590c460e351 |

The images are `testdata/cv/shared/box.png` and `box_in_scene.png`.
`--testdata` and `--models-dir` can override the input locations.

## Reproduce the existing references

From `testdata/dnn`:

```sh
python generate_aliked_lightglue_references.py --resize linear --check
```

`--check` reads the eight existing files and exits with status 1 if any are
missing or differ. Match indices must be identical. Floating-point arrays allow
an absolute difference of `1e-5`, except LightGlue confidence scores (`1e-4`),
which can differ slightly with runtime versions and inference execution.
The C++ ALIKED test also compares coordinates in original pixel space at `1e-4`;
passing this script's comparison does not replace running the C++ tests.

To generate files without replacing the checked-in references:

```sh
python generate_aliked_lightglue_references.py --resize linear --output-dir /tmp/aliked-linear
```

## Regenerate for opencv#30131

[opencv#30131](https://github.com/opencv/opencv/pull/30131) enables
`INTER_LINEAR_EXACT` in ALIKED's `blobFromImage` preprocessing:

```sh
python generate_aliked_lightglue_references.py --resize linear-exact --output-dir /tmp/aliked-linear-exact
```

The script resizes the original **uint8** BGR image with the selected
interpolation, then calls `blobFromImage` without another resize: scale
`1/255`, BGR to RGB, NCHW float32, no crop or padding. Resizing float images
would fall back to ordinary linear interpolation and fail to reproduce the
new preprocessing. Use OpenCV's scaling rather than NumPy division to avoid
introducing an additional rounding difference.

ALIKED outputs are saved as normalized `[-1,1]` keypoints `(N,2)`, row-normalized
descriptors `(N,128)`, and scores `(N,)`, all float32, for each image.
For LightGlue, the script performs the same normalized-to-pixel and
pixel-to-normalized coordinate conversion as the OpenCV 5.x C++ test pipeline.
It saves `matches0` as `lightglue_matches.npy` `(M,2)` int64 and `mscores0` as
`lightglue_mscores.npy` `(M,)` float32.

With the validated environment, ordinary linear interpolation reproduces all
211 existing match pairs; exact linear interpolation produces 209 pairs.
This changes feature ordering as well as values, so regenerate all eight files
together. This script covers the references from #1366; DISK references require
their own generator.

To update checked-in references, run with the corresponding `--resize` and omit
`--output-dir`. Only do this for an OpenCV branch using the same interpolation.

## Validate with the C++ ORT engine

Build OpenCV with `WITH_ONNXRUNTIME=ON` and run its paired test executable:

```sh
OPENCV_FORCE_DNN_ENGINE=2 OPENCV_TEST_DATA_PATH=/path/to/opencv_extra/testdata \
    /path/to/opencv-build/bin/opencv_test_features \
    --gtest_filter='Features2d_ALIKED.Regression:Features2d_LightGlue.Regression'
```

`2` selects `ENGINE_ORT`. Both tests passed with the regenerated exact-linear
references and a fresh build of opencv#30131 at
`dc42d3551150256d319731cbee05ddb589ea27a7` (OpenCV 5.1.0-dev, ONNX Runtime 1.25.1).

At that commit, the default `ENGINE_AUTO` selects `ENGINE_OPENCV`, whose outputs
can differ from ORT in keypoint order even with identical input blobs. In the
same validation, the native engine exchanged rows 661/662 and 953/954 for `box`,
and rows 630/631 for `box_in_scene`. The two tests therefore fail their ordered
comparisons under the default engine. After accounting for those permutations,
the maximum normalized-coordinate difference was `2.39e-7` and the maximum
descriptor difference was `4.42e-6`. This is a separate cross-engine ordering
issue; exact image interpolation does not make network inference bit-exact.
