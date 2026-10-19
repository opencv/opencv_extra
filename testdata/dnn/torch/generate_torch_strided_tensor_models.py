#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Regenerate the Torch7 strided-tensor importer models without dependencies.

Run ``python3 testdata/dnn/torch/generate_torch_strided_tensor_models.py``
from the opencv_extra root. ``--out DIR`` selects another output directory.

Each file is one binary Torch7 FloatTensor record in the layout
cv::dnn::readTorchBlob() parses: TYPE_TORCH, object index, class name,
ndims, sizes, strides, 1-based storage offset, then the FloatStorage object
(TYPE_TORCH, index, class name, element count, float32 data). Integers are
little-endian; the storage holds 0, 1, 2, ... so a read past it is visible.
"""

import argparse
import hashlib
import struct
from pathlib import Path

TYPE_TORCH = 4


def class_name(name):
    raw = name.encode('ascii')
    return struct.pack('<i', len(raw)) + raw


def float_tensor(sizes, strides, offset1, storage_elems):
    assert len(sizes) == len(strides)
    out = struct.pack('<ii', TYPE_TORCH, 1) + class_name('torch.FloatTensor')
    out += struct.pack('<i', len(sizes))
    out += b''.join(struct.pack('<q', v) for v in sizes)
    out += b''.join(struct.pack('<q', v) for v in strides)
    out += struct.pack('<q', offset1)
    out += struct.pack('<ii', TYPE_TORCH, 2) + class_name('torch.FloatStorage')
    out += struct.pack('<q', storage_elems)
    out += b''.join(struct.pack('<f', float(i)) for i in range(storage_elems))
    return out


def cases():
    return [
        ('strided_tensor_within_storage.t7',
         'sizes [1,16], strides [16,1], offset 1, 16-element storage: the view spans exactly the storage',
         float_tensor([1, 16], [16, 1], 1, 16)),
        ('strided_tensor_padded_rows_within_storage.t7',
         'sizes [2,3], strides [4,1], offset 1, 8-element storage: padded rows, a non-contiguous view',
         float_tensor([2, 3], [4, 1], 1, 8)),
        ('strided_tensor_zero_stride.t7',
         'sizes [4,4], strides [0,1], offset 1, 4-element storage: stride 0, which cv::Mat cannot express, rejected',
         float_tensor([4, 4], [0, 1], 1, 4)),
        ('strided_tensor_exceeding_storage.t7',
         'sizes [1,64], strides [1,1], offset 1, 16-element storage: dim 0 fits, the view spans 64 elements',
         float_tensor([1, 64], [1, 1], 1, 16)),
        ('strided_tensor_offset_exceeding_storage.t7',
         'sizes [1,8], strides [7,1], offset 10, 16-element storage: dim 0 fits, the view ends one past the storage',
         float_tensor([1, 8], [7, 1], 10, 16)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, _, blob in cases():
        (args.out / filename).write_bytes(blob)
        print('%s  %s' % (hashlib.sha256(blob).hexdigest(), filename))


if __name__ == '__main__':
    main()
