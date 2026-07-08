#!/usr/bin/env python3
"""
Convert TensorFlow frozen .pb models to ONNX.

Usage:
    python pb_to_onnx.py <model.pb> [output.onnx]

If output path is omitted, uses the path from SPECS.
Called automatically by download_models.py after .pb extraction.
"""

import subprocess
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    onnx = None


# Conversion specs keyed by .pb filename.
# inputs:         tf2onnx --inputs  argument  (name[batch,h,w,c])
# outputs:        tf2onnx --outputs argument  (comma-separated tensor names)
# inputs_as_nchw: tensor name to transpose NHWC→NCHW at model entry (for OpenCV's NCHW blobs)
# output:         default ONNX output path relative to testdata/dnn/
# strip_output_colon_zero: strip trailing :0 from graph output names after conversion
#                          (needed for LapSRN because OpenCV's net.forward() does exact name matching)
SPECS = {
    'tensorflow_inception_graph.pb': {
        'output': 'onnx/models/tensorflow_inception_graph.onnx',
        'inputs': 'input:0[1,224,224,3]',
        'outputs': 'softmax2:0',
        'inputs_as_nchw': 'input:0',
    },
    'ssd_mobilenet_v1_coco_2017_11_17.pb': {
        'output': 'onnx/models/ssd_mobilenet_v1_coco.onnx',
        'inputs': 'image_tensor:0[1,300,300,3]',
        'outputs': 'num_detections:0,detection_boxes:0,detection_scores:0',
        'inputs_as_nchw': 'image_tensor:0',
    },
    'ssd_mobilenet_v2_coco_2018_03_29.pb': {
        'output': 'onnx/models/ssd_mobilenet_v2_coco_2018_03_29.onnx',
        'inputs': 'image_tensor:0[1,300,300,3]',
        'outputs': 'num_detections:0,detection_boxes:0,detection_scores:0',
        'inputs_as_nchw': 'image_tensor:0',
    },
    'ssd_inception_v2_coco_2017_11_17.pb': {
        'output': 'onnx/models/ssd_inception_v2_coco_2017_11_17.onnx',
        'inputs': 'image_tensor:0[1,300,300,3]',
        'outputs': 'num_detections:0,detection_boxes:0,detection_scores:0',
        'inputs_as_nchw': 'image_tensor:0',
    },
    'frozen_east_text_detection.pb': {
        'output': 'onnx/models/east_text_detection.onnx',
        'inputs': 'input_images:0[1,320,320,3]',
        'outputs': 'feature_fusion/Conv_7/Sigmoid:0,feature_fusion/concat_3:0',
        'inputs_as_nchw': 'input_images:0',
    },
    # dnn_superres models — no inputs_as_nchw because dnn_superres.cpp feeds NHWC blobs directly
    'ESPCN_x2.pb': {
        'output': '../../cv/dnn_superres/ESPCN_x2.onnx',
        'inputs': 'IteratorGetNext:0[1,-1,-1,1]',
        'outputs': 'NCHW_output:0',
    },
    'FSRCNN_x2.pb': {
        'output': '../../cv/dnn_superres/FSRCNN_x2.onnx',
        'inputs': 'IteratorGetNext:0[1,-1,-1,1]',
        'outputs': 'NCHW_output:0',
    },
    'FSRCNN_x3.pb': {
        'output': '../../cv/dnn_superres/FSRCNN_x3.onnx',
        'inputs': 'IteratorGetNext:0[1,-1,-1,1]',
        'outputs': 'NCHW_output:0',
    },
    # LapSRN: after conversion strip :0 from output names so OpenCV net.forward() can find them
    'LapSRN_x4.pb': {
        'output': '../../cv/dnn_superres/LapSRN_x4.onnx',
        'inputs': 'IteratorGetNext:0[1,-1,-1,1]',
        'outputs': 'NCHW_output_2x:0,NCHW_output_4x:0',
        'strip_output_colon_zero': True,
    },
}


def strip_output_colon_zero(onnx_path):
    """Strip trailing :0 from graph output names and all references in nodes."""
    if onnx is None:
        print('  [convert] onnx package not installed — skipping :0 strip')
        return
    m = onnx.load(str(onnx_path))
    rename = {o.name: o.name[:-2] for o in m.graph.output if o.name.endswith(':0')}
    if not rename:
        return
    for out in m.graph.output:
        if out.name in rename:
            out.name = rename[out.name]
    for node in m.graph.node:
        node.output[:] = [rename.get(o, o) for o in node.output]
        node.input[:]  = [rename.get(i, i) for i in node.input]
    onnx.save(m, str(onnx_path))
    print('  [convert] stripped :0 from outputs: {}'.format(list(rename.keys())))


def convert(pb_path, onnx_path=None):
    """
    Convert a TF frozen .pb to ONNX using specs from SPECS dict.

    pb_path:   Path to the .pb file (string or Path)
    onnx_path: Override the output path from SPECS (optional)
    Returns True on success, False on failure.
    """
    pb_path = Path(pb_path)
    pb_name = pb_path.name

    if pb_name not in SPECS:
        print('  [convert] no spec for {} — skipping'.format(pb_name))
        return False

    spec = SPECS[pb_name]

    if onnx_path is None:
        onnx_path = pb_path.parent / spec['output']
    onnx_path = Path(onnx_path)

    if onnx_path.exists():
        print('  [convert] {} already exists — skipping'.format(onnx_path))
        return True

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, '-m', 'tf2onnx.convert',
        '--graphdef', str(pb_path),
        '--output',   str(onnx_path),
        '--inputs',   spec['inputs'],
        '--outputs',  spec['outputs'],
    ]
    if spec.get('inputs_as_nchw'):
        cmd += ['--inputs-as-nchw', spec['inputs_as_nchw']]

    print('  [convert] {} -> {}'.format(pb_name, onnx_path))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print('  [convert] tf2onnx failed: {}'.format(e))
        return False

    if spec.get('strip_output_colon_zero'):
        strip_output_colon_zero(onnx_path)

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: pb_to_onnx.py <model.pb> [output.onnx]')
        print('Known models:', list(SPECS.keys()))
        sys.exit(1)

    pb = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    success = convert(pb, out)
    sys.exit(0 if success else 1)
