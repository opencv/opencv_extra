import sys
import os
import argparse
import hashlib

# Only use sys.path if caffe2onnx isn't installed in your pip/conda env
sys.path.insert(0, "/path/to/cloned/caffe2onnx")

def compute_sha256(filepath):
    """Compute SHA-256 hash of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def convert_caffe_to_onnx(prototxt, caffemodel, onnx_path):
    """
    Convert a single Caffe model (.prototxt + .caffemodel) to ONNX.

    Parameters
    ----------
    prototxt   : str — path to Caffe .prototxt (network architecture)
    caffemodel : str — path to Caffe .caffemodel (weights)
    onnx_path  : str — output .onnx file path
    """
    from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
    from caffe2onnx.src.caffe2onnx import Caffe2Onnx

    assert os.path.isfile(prototxt), f"Prototxt not found: {prototxt}"
    assert os.path.isfile(caffemodel), f"Caffemodel not found: {caffemodel}"

    print(f"Converting: {prototxt} -> {onnx_path}")

    graph, params = loadcaffemodel(prototxt, caffemodel)
    converter = Caffe2Onnx(graph, params, onnx_path)
    onnx_model = converter.createOnnxModel()
    saveonnxmodel(onnx_model, onnx_path)

    sha = compute_sha256(onnx_path)
    print(f"  Saved: {onnx_path} (SHA-256: {sha})")
    return sha

# ---------------------------------------------------------------------------
# Model definitions: name -> (prototxt_filename, caffemodel_filename)
# ---------------------------------------------------------------------------
MODELS = {
    "openpose_pose_mpi": (
        "/media/user/hugeDrive2/Open_CV/opencv-5.x/opencv_extra/testdata/dnn/openpose_pose_mpi.prototxt",
        "/media/user/hugeDrive2/Open_CV/opencv-5.x/opencv_extra/testdata/dnn/openpose_pose_mpi.caffemodel"
    ),
    "openpose_pose_mpi_faster_4_stages": (
        "/media/user/hugeDrive2/Open_CV/opencv-5.x/opencv_extra/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt",
        "/media/user/hugeDrive2/Open_CV/opencv-5.x/opencv_extra/testdata/dnn/openpose_pose_mpi.caffemodel"
    ),
    "openpose_pose_coco": (
        "/media/user/hugeDrive2/Open_CV/opencv-5.x/opencv_extra/testdata/dnn/openpose_pose_coco.prototxt",
        "/media/user/hugeDrive2/Open_CV/opencv-5.x/opencv_extra/testdata/dnn/openpose_pose_coco.caffemodel"
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OpenPose Caffe models to ONNX.")
    parser.add_argument("--output_dir", default=".",
                        help="Directory for output ONNX files (default: cwd)")
    args = parser.parse_args()

    try:
        from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
        from caffe2onnx.src.caffe2onnx import Caffe2Onnx
    except ImportError:
        print("Error: caffe2onnx is not installed or not found in sys.path.")
        print("  Make sure the sys.path.insert directory is correct.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("OpenPose Caffe -> ONNX Conversion")
    print("  Output : {}".format(args.output_dir))
    print("=" * 60)

    for name, (proto_path, caffe_path) in MODELS.items():
        # Using just .onnx here, but you can append "(hf-test).onnx" if you still want that suffix
        onnx_path = os.path.join(args.output_dir, name + ".onnx")

        # We don't need os.path.join with input_dir here because the paths in MODELS are hardcoded to the workspace
        convert_caffe_to_onnx(proto_path, caffe_path, onnx_path)

    print("=" * 60)
    print("Done.")
