#!/usr/bin/env python3

from __future__ import print_function
import hashlib
import os
import sys
import tarfile
import shutil
import argparse
import time
from urllib.parse import urlparse
from pathlib import Path
try:
    import requests
except ImportError:
    print("This script requires 'requests' library")
    exit(13)


class BuiltinDownloader:
    MB = 1024*1024
    BUFSIZE = 10*MB
    TIMEOUT = 60

    def print_response(self, response):
        rcode = response.status_code
        rsize = int(response.headers.get('content-length', 0)) / self.MB
        print('    {} [{:.2f} Mb]'.format(rcode, rsize))

    def make_response(self, url, session):
        pieces = urlparse(url)
        if pieces.netloc in ["docs.google.com", "drive.google.com"]:
            return session.get(url, params={'confirm': True}, stream=True, timeout=self.TIMEOUT)
        else:
            return session.get(url, stream=True, timeout=self.TIMEOUT)

    def download_response(self, response, filename):
        with open(filename, 'wb') as f:
            print('    progress ', end='')
            sys.stdout.flush()
            for buf in response.iter_content(self.BUFSIZE):
                if not buf:
                    continue
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()
        print('')

    def download(self, url, filename):
        try:
            session = requests.Session()
            response = self.make_response(url, session)
            self.print_response(response)
            response.raise_for_status()
            self.download_response(response, filename)
            return True
        except Exception as e:
            print('  download failed: {}'.format(e))
            return False


class BuiltinVerifier:
    MB = 1024*1024
    BUFSIZE = 100*MB

    def verify(self, filename, expected_sum):
        if not filename.is_file():
            return False
        sha_calculator = hashlib.sha1()
        try:
            with open(filename, 'rb') as f:
                while True:
                    buf = f.read(self.BUFSIZE)
                    if not buf:
                        break
                    sha_calculator.update(buf)
            if expected_sum != sha_calculator.hexdigest():
                print('  checksum mismatch:')
                print('    expect {}'.format(expected_sum))
                print('    actual {}'.format(sha_calculator.hexdigest()))
                return False
            return True
        except Exception as e:
            print('  verify failed: {}'.format(e))
            return False


class BuiltinExtractor:
    MB = 1024*1024
    BUFSIZE = 100*MB

    def extract(self, arch, member, filename):
        if not arch.is_file():
            return False
        try:
            with tarfile.open(arch) as f:
                if member not in f.getnames():
                    print('  extract - missing member: {}'.format(member))
                    return False
                r = f.extractfile(member)
                with open(filename, 'wb') as f:
                    # print('    progress ', end='')
                    sys.stdout.flush()
                    while True:
                        buf = r.read(self.BUFSIZE)
                        if not buf:
                            break
                        f.write(buf)
                        # print('>', end='')
                        sys.stdout.flush()
            # print('')
            return True
        except Exception as e:
            print('  extract failed: {}'.format(e))
            return False


class Processor:
    def __init__(self, **kwargs):
        self.reference = kwargs.pop('reference', None)
        self.verifier = BuiltinVerifier()
        self.downloader = BuiltinDownloader()
        self.extractor = BuiltinExtractor()

    def prepare_folder(self, filename):
        filename.parent.mkdir(parents=True, exist_ok=True)

    def download(self, url, filename):
        return self.downloader.download(url, filename)

    def verify(self, mdl):
        return self.verifier.verify(mdl.filename, mdl.sha)

    def extract(self, arch, mdl):
        return self.extractor.extract(arch, mdl.member, mdl.filename)

    def ref_copy(self, mdl):
        if not self.reference:
            return False
        candidate = self.reference / mdl.filename
        if not candidate.is_file():
            return False
        print('  ref {} -> {}'.format(candidate, mdl.filename))
        try:
            if candidate.absolute() != mdl.filename.absolute():
                self.prepare_folder(mdl.filename)
                shutil.copy(candidate, mdl.filename)
            if self.verify(mdl):
                return True
            else:
                print('  ref - hash mismatch, removing')
                mdl.filename.unlink()
                return False
        except Exception as e:
            print('  ref failed: {}'.format(e))

    def cleanup(self, filename):
        print("  cleanup - {}".format(filename))
        try:
            filename.unlink()
        except Exception as e:
            print("  cleanup failed: {}".format(e))

    def handle_bad_download(self, filename):
        # rename file for further investigation
        rename_target = filename.with_suffix(filename.suffix + '.invalid')
        print('  renaming invalid file to {}'.format(rename_target))
        try:
            if rename_target.is_file():  # avoid FileExistsError on Windows from os.rename()
                rename_target.unlink()
            filename.rename(rename_target)
        except Exception as e:
            print('  rename failed: {}'.format(e))

    def get_sub(self, arch, mdl):
        print('** {}'.format(mdl.filename))
        if self.verify(mdl):
            return True
        if self.ref_copy(mdl):
            return True
        self.prepare_folder(mdl.filename)
        return self.extract(arch, mdl) and self.verify(mdl)

    def get(self, mdl):
        print("* {}".format(mdl.name))

        # Sub elements - first attempt (ref)
        if len(mdl.sub) > 0:
            if all(self.get_sub(mdl.filename, m) for m in mdl.sub):
                return True

        # File - exists or get from ref or download from internet
        verified = False
        if self.verify(mdl) or self.ref_copy(mdl):
            verified = True

        if not verified:
            self.prepare_folder(mdl.filename)
            for one_url in mdl.url:
                print('  get {}'.format(one_url))
                if self.download(one_url, mdl.filename):
                    if self.verify(mdl):
                        verified = True
                        break
            # TODO: we lose all failed files except the last one
            if not verified and mdl.filename.is_file():
                self.handle_bad_download(mdl.filename)

        if verified or self.verify(mdl):
            # Sub elements - second attempt (extract)
            if len(mdl.sub) > 0:
                return all(self.get_sub(mdl.filename, m) for m in mdl.sub)
            else:
                return True
        else:
            return False


class Model:

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', None)
        self.url = kwargs.pop('url', [])
        self.filename = Path(kwargs.pop('filename'))
        self.sha = kwargs.pop('sha', None)
        self.member = kwargs.pop('member', None)
        self.sub = kwargs.pop('sub', [])
        if not isinstance(self.url, list) and self.url:
            self.url = [self.url]
        # TODO: add completeness assertion

    def __str__(self):
        return 'Model <{}>'.format(self.name)

    def is_archive(self):
        return self.filename.is_file() and ".tar" in self.filename.suffixes


models = [
    Model(
        name='GoogleNet',
        url='http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel',
        sha='405fc5acd08a3bb12de8ee5e23a96bec22f08204',
        filename='bvlc_googlenet.caffemodel'),
    Model(
        name='Alexnet',
        url='http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
        sha='9116a64c0fbe4459d18f4bb6b56d647b63920377',
        filename='bvlc_alexnet.caffemodel'),
    Model(
        name='Inception',
        url='https://github.com/petewarden/tf_ios_makefile_example/raw/master/data/tensorflow_inception_graph.pb',
        sha='c8a5a000ee8d8dd75886f152a50a9c5b53d726a5',
        filename='tensorflow_inception_graph.pb'),
    Model(
        name='Enet',  # https://github.com/e-lab/ENet-training
        url='https://www.dropbox.com/s/tdde0mawbi5dugq/Enet-model-best.net?dl=1',
        sha='b4123a73bf464b9ebe9cfc4ab9c2d5c72b161315',
        filename='Enet-model-best.net'),
    Model(
        name='Fcn',
        url='http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel',
        sha='c449ea74dd7d83751d1357d6a8c323fcf4038962',
        filename='fcn8s-heavy-pascal.caffemodel'),
    Model(
        name='Fcn',
        url='https://github.com/onnx/models/raw/491ce05590abb7551d7fae43c067c060eeb575a6/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx',
        sha='1bb0c7e0034038969aecc6251166f1612a139230',
        filename='onnx/models/fcn-resnet50-12.onnx'),
    Model(
        name='Ssd_vgg16',
        url='https://www.dropbox.com/s/8apyk3uzk2vl522/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel?dl=1',
        sha='0fc294d5257f3e0c8a3c5acaa1b1f6a9b0b6ade0',
        filename='VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel'),
    Model(
        name='ResNet50',
        url=[
            'https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117895&authkey=%21AAFW2%2DFVoxeVRck',
            'https://dl.opencv.org/models/ResNet-50-model.caffemodel'
        ],
        sha='b7c79ccc21ad0479cddc0dd78b1d20c4d722908d',
        filename='ResNet-50-model.caffemodel'),
    Model(
        name='SqueezeNet_v1.1',
        url='https://raw.githubusercontent.com/DeepScale/SqueezeNet/b5c3f1a23713c8b3fd7b801d229f6b04c64374a5/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel',
        sha='3397f026368a45ae236403ccc81cfcbe8ebe1bd0',
        filename='squeezenet_v1.1.caffemodel'),
    Model(
        name='MobileNet-SSD (caffemodel)',  # https://github.com/chuanqi305/MobileNet-SSD
        url='https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/mobilenet_iter_73000.caffemodel',
        sha='19e3ec38842f3e68b02c07a1c24424a1e9db57e9',
        filename='MobileNetSSD_deploy_19e3ec3.caffemodel'),
    Model(
        name='MobileNet-SSD (prototxt)',
        url='https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/97406996b1eee2d40eb0a00ae567cf41e23369f9/deploy.prototxt',
        sha='50cf80235a8fcccc641bf9f8efc803edbf21c615',
        filename='MobileNetSSD_deploy_19e3ec3.prototxt'),
    Model(
        name='OpenFace',  # https://github.com/cmusatyalab/openface
        url='https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7',
        sha='ac8161a4376fb5a79ceec55d85bbb57ef81da9fe',
        filename='openface_nn4.small2.v1.t7'),
    Model(
        name='YoloV2voc',  # https://pjreddie.com/darknet/yolo/
        url='https://pjreddie.com/media/files/yolo-voc.weights',
        sha='1cc1a7f8ad12d563d85b76e9de025dc28ac397bb',
        filename='yolo-voc.weights'),
    Model(
        name='TinyYoloV2voc',  # https://pjreddie.com/darknet/yolo/
        url='https://pjreddie.com/media/files/yolov2-tiny-voc.weights',
        sha='24b4bd049fc4fa5f5e95f684a8967e65c625dff9',
        filename='tiny-yolo-voc.weights'),
    Model(
        name='DenseNet-121 (caffemodel)',  # https://github.com/shicai/DenseNet-Caffe
        url='https://drive.google.com/uc?export=download&id=0B7ubpZO7HnlCcHlfNmJkU2VPelE',
        sha='02b520138e8a73c94473b05879978018fefe947b',
        filename='DenseNet_121.caffemodel'),
    Model(
        name='DenseNet-121 (prototxt)',
        url='https://raw.githubusercontent.com/shicai/DenseNet-Caffe/master/DenseNet_121.prototxt',
        sha='4922099342af5993d9d09f63081c8a392f3c1cc6',
        filename='DenseNet_121.prototxt'),
    Model(
        name='Fast-Neural-Style (starry night)',
        url=[
            'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/starry_night.t7',
            'https://dl.opencv.org/models/fast_neural_style_eccv16_starry_night.t7'
        ],
        sha='5b5e115253197b84d6c6ece1dafe6c15d7105ca6',
        filename='fast_neural_style_eccv16_starry_night.t7'),
    Model(
        name='Fast-Neural-Style (feathers)',
        url=[
            'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/feathers.t7',
            'https://dl.opencv.org/models/fast_neural_style_instance_norm_feathers.t7'
        ],
        sha='9838007df750d483b5b5e90b92d76e8ada5a31c0',
        filename='fast_neural_style_instance_norm_feathers.t7'),
    Model(
        name='MobileNet-SSD (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz',
        sha='a88a18cca9fe4f9e496d73b8548bfd157ad286e2',
        filename='ssd_mobilenet_v1_coco_11_06_217.tar.gz',
        sub=[
            Model(
                member='ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb',
                filename='ssd_mobilenet_v1_coco.pb',
                sha='aaf36f068fab10359eadea0bc68388d96cf68139'
            )
        ]),
    Model(
        name='MobileNet-SSD v1 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
        sha='6157ddb6da55db2da89dd561eceb7f944928e317',
        filename='ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
        sub=[
            Model(
                member='ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb',
                sha='9e4bcdd98f4c6572747679e4ce570de4f03a70e2',
                filename='ssd_mobilenet_v1_coco_2017_11_17.pb'
            )
        ]),
    Model(
        name='MobileNet-SSD v2 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        sha='69c93d29e292bc9682396a5c78355b1dfe481b61',
        filename='ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        sub=[
            Model(
                member='ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
                sha='35d571ac314f1d32ae678a857f87cc0ef6b220e8',
                filename='ssd_mobilenet_v2_coco_2018_03_29.pb')
        ]),
    Model(
        name='Colorization (prototxt)',
        url='https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt',
        sha='f528334e386a69cbaaf237a7611d833bef8e5219',
        filename='colorization_deploy_v2.prototxt'),
    Model(
        name='Colorization (caffemodel)',
        url=[
            'http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel',
            'https://dl.opencv.org/models/colorization_release_v2.caffemodel'
        ],
        sha='21e61293a3fa6747308171c11b6dd18a68a26e7f',
        filename='colorization_release_v2.caffemodel'),
    Model(
        name='Face_detector',
        url='https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
        sha='15aa726b4d46d9f023526d85537db81cbc8dd566',
        filename='opencv_face_detector.caffemodel'),
    Model(
        name='Face_detector (FP16)',
        url='https://github.com/opencv/opencv_3rdparty/raw/19512576c112aa2c7b6328cb0e8d589a4a90a26d/res10_300x300_ssd_iter_140000_fp16.caffemodel',
        sha='31fc22bfdd907567a04bb45b7cfad29966caddc1',
        filename='opencv_face_detector_fp16.caffemodel'),
    Model(
        name='Face_detector (UINT8)',
        url='https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb',
        sha='4f2fdf6f231d759d7bbdb94353c5a68690f3d2ae',
        filename='opencv_face_detector_uint8.pb'),
    Model(
        name='InceptionV2-SSD (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz',
        sha='b9546dcd1ba99282b5bfa81c460008c885ca591b',
        filename='ssd_inception_v2_coco_2017_11_17.tar.gz',
        sub=[
            Model(
                member='ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb',
                sha='554a75594e9fd1ccee291b3ba3f1190b868a54c9',
                filename='ssd_inception_v2_coco_2017_11_17.pb')
        ]),
    Model(
        name='Faster-RCNN',  # https://github.com/rbgirshick/py-faster-rcnn
        url=[
            'https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0',
            'https://dl.opencv.org/models/faster_rcnn_models.tgz'
        ],
        sha='51bca62727c3fe5d14b66e9331373c1e297df7d1',
        filename='faster_rcnn_models.tgz',
        sub=[
            Model(
                member='faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel',
                sha='dd099979468aafba21f3952718a9ceffc7e57699',
                filename='VGG16_faster_rcnn_final.caffemodel'),
            Model(
                member='faster_rcnn_models/ZF_faster_rcnn_final.caffemodel',
                sha='7af886686f149622ed7a41c08b96743c9f4130f5',
                filename='ZF_faster_rcnn_final.caffemodel'),
        ]),
    Model(
        name='R-FCN',  # https://github.com/YuwenXiong/py-R-FCN
        url=[
            'https://onedrive.live.com/download?cid=10B28C0E28BF7B83&resid=10B28C0E28BF7B83%215317&authkey=%21AIeljruhoLuail8',
            'https://dl.opencv.org/models/rfcn_models.tar.gz'
        ],
        sha='bb3180da68b2b71494f8d3eb8f51b2d47467da3e',
        filename='rfcn_models.tar.gz',
        sub=[
            Model(
                member='rfcn_models/resnet50_rfcn_final.caffemodel',
                sha='e00beca7af2790801efb1724d77bddba89e7081c',
                filename='resnet50_rfcn_final.caffemodel'),
        ]),
    Model(
        name='OpenPose/pose/coco',  # https://github.com/CMU-Perceptual-Computing-Lab/openpose
        url=[
            'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel',
            'https://dl.opencv.org/models/openpose_pose_coco.caffemodel'
        ],
        sha='ac7e97da66f3ab8169af2e601384c144e23a95c1',
        filename='openpose_pose_coco.caffemodel'),
    Model(
        name='OpenPose/pose/mpi',  # https://github.com/CMU-Perceptual-Computing-Lab/openpose
        url=[
            'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel',
            'https://dl.opencv.org/models/openpose_pose_mpi.caffemodel'
        ],
        sha='a344f4da6b52892e44a0ca8a4c68ee605fc611cf',
        filename='openpose_pose_mpi.caffemodel'),
    Model(
        name='YOLOv3',  # https://pjreddie.com/darknet/yolo/
        url='https://pjreddie.com/media/files/yolov3.weights',
        sha='520878f12e97cf820529daea502acca380f1cb8e',
        filename='yolov3.weights'),
    Model(
        name='EAST',  # https://github.com/argman/EAST (a TensorFlow model), https://arxiv.org/abs/1704.03155v2 (a paper)
        url='https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1',
        sha='3ca8233d6edd748f7ed23246c8ca24cbf696bb94',
        filename='frozen_east_text_detection.tar.gz',
        sub=[
            Model(
                member='frozen_east_text_detection.pb',
                sha='fffabf5ac36f37bddf68e34e84b45f5c4247ed06',
                filename='frozen_east_text_detection.pb'),
        ]),
    Model(
        name='Faster-RCNN, InveptionV2 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        sha='c710f25e5c6a3ce85fe793d5bf266d581ab1c230',
        filename='faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        sub=[
            Model(
                member='faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
                sha='f2e4bf386b9bb3e25ddfcbbd382c20f417e444f3',
                filename='faster_rcnn_inception_v2_coco_2018_01_28.pb'),
        ]),
    Model(
        name='ssd_mobilenet_v1_ppn_coco (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
        sha='549ae0fd82c202786abe53c306b191c578599c44',
        filename='ssd_mobilenet_v1_ppn_coco.tar.gz',
        sub=[
            Model(
                member='ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb',
                sha='7943c51c6305b38173797d4afbf70697cf57ab48',
                filename='ssd_mobilenet_v1_ppn_coco.pb'),
        ]),
    Model(
        name='mask_rcnn_inception_v2_coco_2018_01_28 (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        sha='f8a920756744d0f7ee812b3ec2474979f74ab40c',
        filename='mask_rcnn_inception_v2_coco_2018_01_28.tar.gz',
        sub=[
            Model(
                member='mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
                sha='c8adff66a1e23e607f57cf1a7cfabad0faa371f9',
                filename='mask_rcnn_inception_v2_coco_2018_01_28.pb'),
        ]),
    Model(
        name='faster_rcnn_resnet50_coco (TensorFlow)',
        url='http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        sha='3066e8dd156b99c4b4d78a2ccd13e33fc263beb7',
        filename='faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        sub=[
            Model(
                member='faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb',
                sha='27feaef9924650299b2ef5d29f041627b6f298b2',
                filename='faster_rcnn_resnet50_coco_2018_01_28.pb'),
        ]),
    Model(
        name='AlexNet (ONNX)',
        url='https://github.com/onnx/models/raw/69c5d3751dda5349fd3fc53f525395d180420c07/vision/classification/alexnet/model/bvlcalexnet-8.onnx',
        sha='b256703f2b125d8681a0a6e5a40a6c9deb7d2b4b',
        filename='onnx/models/alexnet.onnx'),
    Model(
        name='GoogleNet (ONNX)',
        url='https://github.com/onnx/models/raw/69c5d3751dda5349fd3fc53f525395d180420c07/vision/classification/inception_and_googlenet/googlenet/model/googlenet-8.onnx',
        sha='534a16d7e2472f6a9a1925a5ee6c9abc2f5c02b0',
        filename='onnx/models/googlenet.onnx'),
    Model(
        name='CaffeNet (ONNX)',
        url='https://github.com/onnx/models/raw/4eff8f9b9189672de28d087684e7085ad977747c/vision/classification/caffenet/model/caffenet-8.tar.gz',
        sha='f9f5dd60d4c9172a7e26bd4268eab7ecddb37393',
        filename='bvlc_reference_caffenet.tar.gz',
        sub=[
            Model(
                member='bvlc_reference_caffenet/model.onnx',
                sha='6b2be0cd598914e13b60787c63cba0533723d746',
                filename='onnx/models/caffenet.onnx'),
            Model(
                member='bvlc_reference_caffenet/test_data_set_0/input_0.pb',
                sha='e5d6fb75a66ef157023a7fc2f88abdcb371f2f16',
                filename='onnx/data/input_caffenet.pb'),
            Model(
                member='bvlc_reference_caffenet/test_data_set_0/output_0.pb',
                sha='eaff902ef71a648aaaeffa495e5fddf2dc0b77c1',
                filename='onnx/data/output_caffenet.pb'),
        ]),
    Model(
        name='RCNN_ILSVRC13 (ONNX)',
        url='https://github.com/onnx/models/raw/cbda9ebd037241c6c6a0826971741d5532af8fa4/vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-8.tar.gz',
        sha='b1b27a41066c26f824d57e99036dc885459017f0',
        filename='bvlc_reference_rcnn_ilsvrc13.tar.gz',
        sub=[
            Model(
                member='bvlc_reference_rcnn_ilsvrc13/model.onnx',
                sha='fbf174b62a1918bff43c0287e41fdc6017b46256',
                filename='onnx/models/rcnn_ilsvrc13.onnx'),
            Model(
                member='bvlc_reference_rcnn_ilsvrc13/test_data_set_0/input_0.pb',
                sha='dcfd587bede888606a7f10e9feadc7f25bed7da4',
                filename='onnx/data/input_rcnn_ilsvrc13.pb'),
            Model(
                member='bvlc_reference_rcnn_ilsvrc13/test_data_set_0/output_0.pb',
                sha='e09eea540b93a2f450e32db59e198ca96c3b8637',
                filename='onnx/data/output_rcnn_ilsvrc13.pb'),
        ]),
    Model(
        name='ZFNet512 (ONNX)',
        url='https://github.com/onnx/models/raw/f884b33c3e2371952aad7ea091898f418c830fe5/vision/classification/zfnet-512/model/zfnet512-8.tar.gz',
        sha='c040c455c8aac71c8cda57595b698b76449e4ff4',
        filename='zfnet512.tar.gz',
        sub=[
            Model(
                member='zfnet512/model.onnx',
                sha='c32b9ae0bbe65e2ee60f98639b170645000e2c75',
                filename='onnx/models/zfnet512.onnx'),
            Model(
                member='zfnet512/test_data_set_0/input_0.pb',
                sha='2dc2c8020edbd84a52f0550d6666c9ae7e93c01f',
                filename='onnx/data/input_zfnet512.pb'),
            Model(
                member='zfnet512/test_data_set_0/output_0.pb',
                sha='a74974096088954ca4e4e89bec212c1ac2ab0745',
                filename='onnx/data/output_zfnet512.pb'),
        ]),
    Model(
        name='VGG16_bn (ONNX)',
        url='https://github.com/onnx/models/raw/f884b33c3e2371952aad7ea091898f418c830fe5/vision/classification/vgg/model/vgg16-bn-7.tar.gz',
        sha='60f4685aed632d2ce3b137017cf44ae1a5c55459',
        filename='vgg16-bn.tar.gz',
        sub=[
            Model(
                member='vgg16-bn/vgg16-bn.onnx',
                sha='e282e2137f1317d03ca1f2702e9cfddaf847e44d',
                filename='onnx/models/vgg16-bn.onnx'),
            Model(
                member='vgg16-bn/test_data_set_0/input_0.pb',
                sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
                filename='onnx/data/input_vgg16-bn.pb'),
            Model(
                member='vgg16-bn/test_data_set_0/output_0.pb',
                sha='418b1a426a2a4105cfd9a77a965ae67dc105891b',
                filename='onnx/data/output_vgg16-bn.pb'),
        ]),
    Model(
        name='ResNet-18v1 (ONNX)',
        url=[
            'https://github.com/onnx/models/raw/69c5d3751dda5349fd3fc53f525395d180420c07/vision/classification/resnet/model/resnet18-v1-7.tar.gz',
            'https://dl.opencv.org/models/resnet18v1.tar.gz'
        ],
        sha='d132be4857d024de9caa21fd5300dee7c063bc35',
        filename='resnet18v1.tar.gz',
        sub=[
            Model(
                member='resnet18v1/resnet18v1.onnx',
                sha='9d96d7142c5ce43aa61ce67124b8eb5530afff4c',
                filename='onnx/models/resnet18v1.onnx'),
            Model(
                member='resnet18v1/test_data_set_0/input_0.pb',
                sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
                filename='onnx/data/input_resnet18v1.pb'),
            Model(
                member='resnet18v1/test_data_set_0/output_0.pb',
                sha='70e0ad583cf922452ac6e52d882b5127db086a45',
                filename='onnx/data/output_resnet18v1.pb'),
        ]),
    Model(
        name='ResNet-50v1 (ONNX)',
        url=[
            'https://github.com/onnx/models/raw/69c5d3751dda5349fd3fc53f525395d180420c07/vision/classification/resnet/model/resnet50-v1-7.tar.gz',
            'https://dl.opencv.org/models/resnet50v1.tar.gz'
        ],
        sha='a4ac2da7e0024d61fdb80481496ba966b48b9fea',
        filename='resnet50v1.tar.gz',
        sub=[
            Model(
                member='resnet50v1/resnet50v1.onnx',
                sha='06aa26c6de448e11c64cd80cf06f5ab01de2ec9b',
                filename='onnx/models/resnet50v1.onnx'),
            Model(
                member='resnet50v1/test_data_set_0/input_0.pb',
                sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
                filename='onnx/data/input_resnet50v1.pb'),
            Model(
                member='resnet50v1/test_data_set_0/output_0.pb',
                sha='40deb324ddba7db4117568e1e3911e7a771fb260',
                filename='onnx/data/output_resnet50v1.pb'),
        ]),
    Model(
        name='ResNet50-Int8 (ONNX)',
        url='https://github.com/onnx/models/raw/771185265efbdc049fb223bd68ab1aeb1aecde76/vision/classification/resnet/model/resnet50-v1-12-int8.tar.gz',
        sha='2ff2a58f4a27362ee6234915452e86287cdcf269',
        filename='resnet50-v1-12-int8.tar.gz',
        sub=[
            Model(
                member='resnet50-v1-12-int8/resnet50-v1-12-int8.onnx',
                sha='5fbeac70e1a3af3253c21e0e4008a784aa61929f',
                filename='onnx/models/resnet50_int8.onnx'),
            Model(
                member='resnet50-v1-12-int8/test_data_set_0/input_0.pb',
                sha='0946521c8afcfea9340390298a41fb11496b3556',
                filename='onnx/data/input_resnet50_int8.pb'),
            Model(
                member='resnet50-v1-12-int8/test_data_set_0/output_0.pb',
                sha='6d45d2f06150e9045631c7928093728b07c8b12d',
                filename='onnx/data/output_resnet50_int8.pb'),
        ]),
    # TODO: bad file
    Model(
        name='ResNet101_DUC_HDC (ONNX)',
        url=[
            'https://github.com/onnx/models/raw/69c5d3751dda5349fd3fc53f525395d180420c07/vision/classification/resnet/model/resnet101-v1-7.tar.gz',
            'https://dl.opencv.org/models/ResNet101_DUC_HDC.tar.gz'
        ],
        sha='f8314f381939d01045ac31dbb53d7d35fe3ff9a0',
        filename='ResNet101_DUC_HDC.tar.gz',
        sub=[
            Model(
                member='ResNet101_DUC_HDC/ResNet101_DUC_HDC.onnx',
                sha='83f9cefdf3606a37dd4901a925bb9116795dae39',
                filename='onnx/models/resnet101_duc_hdc.onnx'),
            Model(
                member='ResNet101_DUC_HDC/test_data_set_0/input_0.pb',
                sha='099d0e32742a2fa6a69c329f1bff699fb7266b33',
                filename='onnx/data/input_resnet101_duc_hdc.pb'),
            Model(
                member='ResNet101_DUC_HDC/test_data_set_0/output_0.pb',
                sha='3713a21bb7228d3179721810bb72565aebee7033',
                filename='onnx/data/output_resnet101_duc_hdc.pb'),
        ]),
    Model(
        name='TinyYolov2 (ONNX)',
        url='https://github.com/onnx/models/raw/3d4b2c28f951064ab35c89d5f5c3ffe74a149e4b/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-1.tar.gz',
        sha='b9102abb8fa6f51368119b52146c30189353164a',
        filename='tiny_yolov2.tar.gz',
        sub=[
            Model(
                member='tiny_yolov2/model.onnx',
                sha='433fecbd32ac8b9be6f5ee10c39dcecf9dc5c151',
                filename='onnx/models/tiny_yolo2.onnx'),
            Model(
                member='tiny_yolov2/test_data_set_0/input_0.pb',
                sha='a0412fde98ca21d726c0c86ef007c11aa4678e3c',
                filename='onnx/data/input_tiny_yolo2.pb'),
            Model(
                member='tiny_yolov2/test_data_set_0/output_0.pb',
                sha='f9be0446cac76fe38bb23cb09ed23c317907f505',
                filename='onnx/data/output_tiny_yolo2.pb'),
        ]),
    Model(
        name='CNN Mnist (ONNX)',
        url='https://github.com/onnx/models/raw/cbda9ebd037241c6c6a0826971741d5532af8fa4/vision/classification/mnist/model/mnist-7.tar.gz',
        sha='8bcd3372e44bd95dc8a211bc31fb3025d8edf9f9',
        filename='mnist.tar.gz',
        sub=[
            Model(
                member='mnist/model.onnx',
                sha='e4fb4914cd1d9e0faed3294e5cecfd1847339763',
                filename='onnx/models/cnn_mnist.onnx'),
            Model(
                member='mnist/test_data_set_0/input_0.pb',
                sha='023f6c94951ab386957964e39727aa43d8c45ea8',
                filename='onnx/data/input_cnn_mnist.pb'),
            Model(
                member='mnist/test_data_set_0/output_0.pb',
                sha='79f3028d97df835b058849d357e06d4c0bfcf5b3',
                filename='onnx/data/output_cnn_mnist.pb'),
        ]),
    Model(
        name='MobileNetv2 (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz',
        sha='7f1429a8e1f3118a05943ff3ed54dbc9eb55691a',
        filename='mobilenetv2-1.0.tar.gz',
        sub=[
            Model(
                member='mobilenetv2-1.0/mobilenetv2-1.0.onnx',
                sha='80c97941c3ce34d05bc3d3c9d6e04c44c15906bc',
                filename='onnx/models/mobilenetv2.onnx'),
            Model(
                member='mobilenetv2-1.0/test_data_set_0/input_0.pb',
                sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
                filename='onnx/data/input_mobilenetv2.pb'),
            Model(
                member='mobilenetv2-1.0/test_data_set_0/output_0.pb',
                sha='7e58c6faca7fc3b844e18364ae92606aa3f0b18e',
                filename='onnx/data/output_mobilenetv2.pb'),
        ]),
    Model(
        name='LResNet100E-IR (ONNX)',
        url='https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz',
        sha='b1178813b705d9d44ed806aa442f0b1cb11aea0a',
        filename='resnet100.tar.gz',
        sub=[
            Model(
                member='resnet100/resnet100.onnx',
                sha='d307e426cf55cddf9f9292b5ffabb474eec93638',
                filename='onnx/models/LResNet100E_IR.onnx'),
            Model(
                member='resnet100/test_data_set_0/input_0.pb',
                sha='d80a849e000907734bd0061ba570f734784f7d38',
                filename='onnx/data/input_LResNet100E_IR.pb'),
            Model(
                member='resnet100/test_data_set_0/output_0.pb',
                sha='f54c73699d00b18b5c40e4ea895b1e88e7f8dea3',
                filename='onnx/data/output_LResNet100E_IR.pb'),
        ]),
    Model(
        name='Emotion FERPlus (ONNX)',
        url='https://github.com/onnx/models/raw/7cee9777a86dd6e80040d6b786869a83d2ad1273/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.tar.gz',
        sha='9ff80899c0cd468999db5d8ffde98780ef85455e',
        filename='emotion_ferplus.tar.gz',
        sub=[
            Model(
                member='emotion_ferplus/model.onnx',
                sha='2ef5b3a6404a5feb8cc396d66c86838c4c750a7e',
                filename='onnx/models/emotion_ferplus.onnx'),
            Model(
                member='emotion_ferplus/test_data_set_0/input_0.pb',
                sha='29621536528116fc12f02bc81c7265f7ffe7c8bb',
                filename='onnx/data/input_emotion_ferplus.pb'),
            Model(
                member='emotion_ferplus/test_data_set_0/output_0.pb',
                sha='54f7892240d2d9298f5a8064a46fc3a8987015a5',
                filename='onnx/data/output_emotion_ferplus.pb'),
        ]),
    Model(
        name='Squeezenet (ONNX)',
        url='https://github.com/onnx/models/raw/f884b33c3e2371952aad7ea091898f418c830fe5/vision/classification/squeezenet/model/squeezenet1.0-8.tar.gz',
        sha='57348321d4d460c07c41af814def3abe728b3a03',
        filename='squeezenet.tar.gz',
        sub=[
            Model(
                member='squeezenet/model.onnx',
                sha='c3f272e672fa64a75fb4a2e48dd2ca25fcc76c49',
                filename='onnx/models/squeezenet.onnx'),
            Model(
                member='squeezenet/test_data_set_0/input_0.pb',
                sha='55c285cfbc4d61e3c026302a3af9e7d220b82d0a',
                filename='onnx/data/input_squeezenet.pb'),
            Model(
                member='squeezenet/test_data_set_0/output_0.pb',
                sha='e4f3c0c989cc7025ca94759492508d8f4ef3287b',
                filename='onnx/data/output_squeezenet.pb'),
        ]),
    Model(
        name='DenseNet121 (ONNX)',
        url='https://github.com/onnx/models/raw/4eff8f9b9189672de28d087684e7085ad977747c/vision/classification/densenet-121/model/densenet-8.tar.gz',
        sha='338b70e871e73b0550fc8ccc0863b8382e90e8e5',
        filename='densenet121.tar.gz',
        sub=[
            Model(
                member='densenet121/model.onnx',
                sha='2874279d0f56f15f4e7e9208526c1b35d85d5ad1',
                filename='onnx/models/densenet121.onnx'),
            Model(
                member='densenet121/test_data_set_0/input_0.pb',
                sha='d6146a5b08a85309a3b8ada313ae5887c2aa7e3e',
                filename='onnx/data/input_densenet121.pb'),
            Model(
                member='densenet121/test_data_set_0/output_0.pb',
                sha='f1fd0d5e8d48aff3df2c5c809ea24e982d72028e',
                filename='onnx/data/output_densenet121.pb'),
        ]),
    Model(
        name='Inception v1 (ONNX)',
        url='https://github.com/onnx/models/raw/4eff8f9b9189672de28d087684e7085ad977747c/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-8.tar.gz',
        sha='94ecb2bd1426704dca578dc746e3c27bedf22352',
        filename='inception_v1.tar.gz',
        sub=[
            Model(
                member='inception_v1/model.onnx',
                sha='f45896d8d35248a62ea551db922d982a90214517',
                filename='onnx/models/inception_v1.onnx'),
            Model(
                member='inception_v1/test_data_set_0/input_0.pb',
                sha='7ec7a82aa2fecd2c875b7b198ecd9a428bc9f462',
                filename='onnx/data/input_inception_v1.pb'),
            Model(
                member='inception_v1/test_data_set_0/output_0.pb',
                sha='870a30306bd2b82d5393a0ff5570b022681ef7b6',
                filename='onnx/data/output_inception_v1.pb'),
        ]),
    Model(
        name='Inception v2 (ONNX)',
        url='https://github.com/onnx/models/raw/4eff8f9b9189672de28d087684e7085ad977747c/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-8.tar.gz',
        sha='d07a442a84d939232c37c976fd8d624fa9f82026',
        filename='inception_v2.tar.gz',
        sub=[
            Model(
                member='inception_v2/model.onnx',
                sha='cfa84f36bcae8910e0875872383991cb0c3b9a80',
                filename='onnx/models/inception_v2.onnx'),
            Model(
                member='inception_v2/test_data_set_0/input_0.pb',
                sha='f4ed6d838c20dbfc3bcf6abfd23c78d74892a5fe',
                filename='onnx/data/input_inception_v2.pb'),
            Model(
                member='inception_v2/test_data_set_0/output_0.pb',
                sha='cb75fb6db82290c49879380ce72c71e17eda76d0',
                filename='onnx/data/output_inception_v2.pb'),
        ]),
    Model(
        name='Shufflenet (ONNX)',
        url='https://github.com/onnx/models/raw/f884b33c3e2371952aad7ea091898f418c830fe5/vision/classification/shufflenet/model/shufflenet-9.tar.gz',
        sha='c99afcb7fcc809c0688cc99cb3709a052fde1de7',
        filename='shufflenet.tar.gz',
        sub=[
            Model(
                member='shufflenet/model.onnx',
                sha='a781faf9f1fe6d001cd7b6b5a7d1a228da0ff17b',
                filename='onnx/models/shufflenet.onnx'),
            Model(
                member='shufflenet/test_data_set_0/input_0.pb',
                sha='27d31be9a084c1d1d1eacbd766f4c43d59352a07',
                filename='onnx/data/input_shufflenet.pb'),
            Model(
                member='shufflenet/test_data_set_0/output_0.pb',
                sha='6a33ed6ccef4c69a27a3993363c3f854d0f79bb0',
                filename='onnx/data/output_shufflenet.pb'),
        ]),
    Model(
        name='ResNet-34_kinetics (ONNX)', # https://github.com/kenshohara/video-classification-3d-cnn-pytorch
        url='https://www.dropbox.com/s/065l4vr8bptzohb/resnet-34_kinetics.onnx?dl=1',
        sha='88897629e4abb0fddef939f0c2d668a4edeb0788',
        filename='resnet-34_kinetics.onnx'),
    Model(
        name='Alexnet Facial Keypoints (ONNX)', # https://github.com/ismalakazel/Facial-Keypoint-Detection
        url='https://drive.google.com/uc?export=dowload&id=1etGXT9WQK1KjDkJ0pUTH-CaHHva4p9cY',
        sha='e1b82b56b59ab96b50189e1b39487d91d4fa0eea',
        filename='onnx/models/facial_keypoints.onnx'),
    Model(
        name='LightWeight Human Pose Estimation (ONNX)', # https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
        url='https://drive.google.com/uc?export=dowload&id=1--Ij_gIzCeNA488u5TA4FqWMMdxBqOji',
        sha='5960f7aef233d75f8f4020be1fd911b2d93fbffc',
        filename='onnx/models/lightweight_pose_estimation_201912.onnx'),
    Model(
        name='EfficientDet-D0', # https://github.com/google/automl
        url='https://www.dropbox.com/s/9mqp99fd2tpuqn6/efficientdet-d0.pb?dl=1',
        sha='f178cc17b44e3ed2f3956a0adc1800a7d2a3b3ae',
        filename='efficientdet-d0.pb'),
    Model(
        name='YOLOv4',  # https://github.com/opencv/opencv/issues/17148
        url="https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
        sha='0143deb6c46fcc7f74dd35bf3c14edc3784e99ee',
        filename='yolov4.weights'),
    Model(
        name='YOLOv4-tiny-2020-12',  # https://github.com/opencv/opencv/issues/17148
        url='https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights',
        sha='451caaab22fb9831aa1a5ee9b5ba74a35ffa5dcb',
        filename='yolov4-tiny-2020-12.weights'),
    Model(
        name='YOLOv4x-mish',  # https://github.com/opencv/opencv/issues/18975
        url='https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights',
        sha='a6f2879af2241de2e9730d317a55db6afd0af00b',
        filename='yolov4x-mish.weights'),
    Model(
        name='GSOC2016-GOTURN',  # https://github.com/opencv/opencv_contrib/issues/941
        url=[
            'https://docs.google.com/uc?export=download&id=1j4UTqVE4EGaUFiK7a5I_CYX7twO9c5br',
            'https://dl.opencv.org/models/goturn.caffemodel'
        ],
        sha='49776d262993c387542f84d9cd16566840404f26',
        filename='gsoc2016-goturn/goturn.caffemodel'),
    Model(
        name='DaSiamRPM Tracker network (ONNX)',
        url='https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=1',
        sha='91b774fce7df4c0e4918469f0f482d9a27d0e2d4',
        filename='onnx/models/dasiamrpn_model.onnx'),
    Model(
        name='DaSiamRPM Tracker kernel_r1 (ONNX)',
        url='https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=1',
        sha='bb64620a54348657133eb28be2d3a2a8c76b84b3',
        filename='onnx/models/dasiamrpn_kernel_r1.onnx'),
    Model(
        name='DaSiamRPM Tracker kernel_cls1 (ONNX)',
        url='https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=1',
        sha='e9ccd270ce8059bdf7ed0d1845c03ef4a951ee0f',
        filename='onnx/models/dasiamrpn_kernel_cls1.onnx'),
    Model(
        name='crnn',
        url='https://drive.google.com/uc?export=dowload&id=1ooaLR-rkTl8jdpGy1DoQs0-X0lQsB6Fj',
        sha='270d92c9ccb670ada2459a25977e8deeaf8380d3',
        filename='onnx/models/crnn.onnx'),
    Model(
        name='DB_TD500_resnet50',
        url='https://drive.google.com/uc?export=dowload&id=19YWhArrNccaoSza0CfkXlA8im4-lAGsR',
        sha='1b4dd21a6baa5e3523156776970895bd3db6960a',
        filename='onnx/models/DB_TD500_resnet50.onnx'),
    Model(
        name='face_recognizer_fast',
        url='https://drive.google.com/uc?export=dowload&id=1ClK9WiB492c5OZFKveF3XiHCejoOxINW',
        sha='12ff8b1f5c8bff62e8dd91eabdacdfc998be255e',
        filename='onnx/models/face_recognizer_fast.onnx'),
    Model(
        name='MobileNetv2 FP16 (ONNX)',
        url='https://github.com/zihaomu/zihaomu/files/9393786/mobilenetv2_fp16_v7.tar.gz',
        sha='018d42b1b1283e6025a0455deffe9f0e9930e839',
        filename='mobilenetv2_fp16_v7.tar.gz',
        sub=[
            Model(
                member='mobilenetv2_fp16_v7/mobilenetv2_fp16.onnx',
                sha='ab9352de8e07b798417922f23e97c8488bd50017',
                filename='onnx/models/mobilenetv2_fp16.onnx'),
            Model(
                member='mobilenetv2_fp16_v7/input_mobilenetv2_fp16.npy',
                sha='cbb97c31abc07ff8c68f5028c634d79f8b83b560',
                filename='onnx/data/input_mobilenetv2_fp16.npy'),
            Model(
                member='mobilenetv2_fp16_v7/output_mobilenetv2_fp16.npy',
                sha='397560616c47b847340cec9561e12a13b29ae32e',
                filename='onnx/data/output_mobilenetv2_fp16.npy'),
        ]),
    Model(
        name='wechat_qr_detect (prototxt)',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt',
        sha='a6936962139282d300ebbf15a54c2aa94b144bb7',
        filename='wechat_2021-01/detect.prototxt'),
    Model(
        name='wechat_qr_detect (caffemodel)',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel',
        sha='d587623a055cbd58a648de62a8c703c7abb05f6d',
        filename='wechat_2021-01/detect.caffemodel'),
    Model(
        name='wechat_super_resolution (prototxt)',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt',
        sha='39e1f1031c842766f1cc126615fea8e8256facd2',
        filename='wechat_2021-01/sr.prototxt'),
    Model(
        name='wechat_super_resolution (caffemodel)',
        url='https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel',
        sha='2b181b55d1d7af718eaca6cabdeb741217b64c73',
        filename='wechat_2021-01/sr.caffemodel'),
    Model(
        name='yolov7',
        url=[
            'https://dl.opencv.org/models/yolov7/yolov7.onnx'
        ],
        sha='9f5199c266418462771a26a7b8ea25a90412ce2e',
        filename='onnx/models/yolov7.onnx'),
    Model(
        name='yolox_s_inf_decoder',
        url=[
            'https://drive.google.com/u/0/uc?id=12dVy3ob7T4fYHOkLYnrpUmlysq6JEc5P&export=download',
            'https://dl.opencv.org/models/yolox/yolox_s_inf_decoder.onnx'
        ],
        sha='b205b00122cc7bf559a0e845680408320df3a898',
        filename='onnx/models/yolox_s_inf_decoder.onnx'),
    Model(
        name='yolov6n',
        url='https://dl.opencv.org/models/yolov6/yolov6n.onnx',
        sha='a704c0ace51103a43920c50a396b2c8b09d2daec',
        filename='onnx/models/yolov6n.onnx'),
    Model(
        name='yolov8n',
        url=[
            'https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx?download=true',
            'https://dl.opencv.org/models/yolov8/yolov8n.onnx'
        ],
        sha='136807b88d0b02bc226bdeb9741141d857752e10',
        filename='onnx/models/yolov8n.onnx'),
    Model(
        name='yolov8x',
        url=[
            'https://huggingface.co/cabelo/yolov8/resolve/main/yolov8x.onnx?download=true',
            'https://dl.opencv.org/models/yolov8/yolov8x.onnx'
        ],
        sha='462f15d668c046d38e27d3df01fe8142dd004cb4',
        filename='onnx/models/yolov8x.onnx'),
    Model(
        name='yolov9t',
        url='https://dl.opencv.org/models/yolov9/yolov9t.onnx',
        sha='330292f15e1b312b11ce58e70a9e455d54415fa3',
        filename='onnx/models/yolov9t.onnx'),
    Model(
        name='yolov10s',
        url='https://dl.opencv.org/models/yolov10/yolov10s.onnx',
        sha='5311212e431912a27d5f54b3a5277bc573890a99',
        filename='onnx/models/yolov10s.onnx'),
    Model(
        name='yolo_nas_s',
        url='https://dl.opencv.org/models/yolo-nas/yolo_nas_s.onnx',
        sha='089942fbdf8591875a7a6ff10ac50fb6864e7aa4',
        filename='onnx/models/yolo_nas_s.onnx'),

    Model(
        name='NanoTrackV1 (ONNX, backbone)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrack/models/nanotrack_backbone_sim.onnx',
        sha='9b083a2dbe10dcfe17e694879aa6749302a5888f',
        filename='onnx/models/nanotrack_backbone_sim.onnx'),
    Model(
        name='NanoTrackV1 (ONNX, head)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrack/models/nanotrack_head_sim.onnx',
        sha='8fa668893b27b726f9cab6695846b4690650a199',
        filename='onnx/models/nanotrack_head_sim.onnx'),
    Model(
        name='NanoTrackV2 (ONNX, backbone)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrackV2/models/nanotrack_backbone_sim_v2.onnx',
        sha='6e773a364457b78574f9f63a23b0659ee8646f8f',
        filename='onnx/models/nanotrack_backbone_sim_v2.onnx'),
    Model(
        name='NanoTrackV2 (ONNX, head)',
        url='https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrackV2/models/nanotrack_head_sim_v2.onnx',
        sha='39f168489671700cf739e402dfc67d41ce648aef',
        filename='onnx/models/nanotrack_head_sim_v2.onnx'),
    Model(
        name='Face Mesh (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite?generation=1668295060280094',
        sha='eb01d1d88c833aaea64c880506da72e4a4f43154',
        filename='tflite/face_landmark.tflite'),
    Model(
        name='Face Detection (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite?generation=1661875748538815',
        sha='e8f749fafc23bb88daac85bc9f7e0698436f29a0',
        filename='tflite/face_detection_short_range.tflite'),
    Model(
        name='Selfie Segmentation (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite?generation=1661875931201364',
        sha='8d497f51bd678fa5fb95c3871be72eb5d722b831',
        filename='tflite/selfie_segmentation.tflite'),
    Model(
        name='Hair Segmentation (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite?generation=1661875756623461',
        sha='bba28400dfc264b1ed7ee95df718fada1879644d',
        filename='tflite/hair_segmentation.tflite'),
    Model(
        name='YuNet',
        url='https://github.com/ShiqiYu/libfacedetection.train/raw/02246e79b1e976c83d1e135a85e0628120c93769/onnx/yunet_s_640_640.onnx',
        sha='acbe4b5976ade60c4b866a30d0720d71589c8bbc',
        filename='onnx/models/yunet-202303.onnx'),
    Model(
        name='EfficientDet (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/coco_efficientdet_lite0_v1_1.0_quant_2021_09_06.tflite?generation=1661875692679200',
        sha='200217d746d58e68028a64ad0472631060e6affb',
        filename='tflite/coco_efficientdet_lite0_v1_1.0_quant_2021_09_06.tflite'),
    Model(
        name='PP_OCRv3_DB_text_det (ONNX)',
        url='https://github.com/zihaomu/zihaomu/files/9740907/PP_OCRv3_DB_text_det.tar.gz',
        sha='a2a008361d785fbe32a22ec2106621ecd1576f48',
        filename='PP_OCRv3_DB_text_det.tar.gz',
        sub=[
            Model(
                member='PP_OCRv3_DB_text_det/PP_OCRv3_DB_text_det.onnx',
                sha='f541f0b448561c7ad919ba9fffa72ff105062934',
                filename='onnx/models/PP_OCRv3_DB_text_det.onnx'),
        ]),
    Model(
        name='YOLOv5n (ONNX)',
        url='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
        sha='90aa4b5f8e1ba19166f214f3e4258e5d1d05070b',
        filename='yolov5n.onnx'),
    Model(
        # model link form https://github.com/CVHub520/X-AnyLabeling/blob/20dff273cdb0658fdb81cad72aef0c1add33fdb1/docs/models_list.md
        name='YOLOv8n (ONNX)',
        url='https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx',
        sha='68f864475d06e2ec4037181052739f268eeac38d',
        filename='yolov8n.onnx'),
    Model(
        name='YOLOX_S (ONNX)',
        url='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx',
        sha='0249e66522b38462e6962915a8850a5908023a1c',
        filename='yolox_s.onnx'),
    Model(
        # model link from https://github.com/onnx/models/tree/main/vision/classification/efficientnet-lite4
        name='EfficientNet Lite4 (ONNX)',
        url='https://github.com/onnx/models/raw/280606e8de3a8fed89a1ef3031d32032af17744b/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx',
        sha='78c3d1eb329793b253565ef319182f03ee9af78e',
        filename='efficientnet-lite4.onnx'),
    Model(
        name='SFace (ONNX)',
        url='https://github.com/opencv/opencv_zoo/raw/ba91a3b91d00d76e86540d4013f944bd6b514e39/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
        sha='316ca25772af10f61e356f81f0ec68caf6909a51',
        filename='face_recognition_sface_2021dec.onnx'),
    Model(
        name='MediaPipe Palm Detector (ONNX)',
        url='https://github.com/opencv/opencv_zoo/raw/8de36535ea29e8f9d41e6e3fa5a0df14bab00ec5/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx',
        sha='b9e6df1d4f93ee1b0b4f5c99a2f88716ccd7ca9a',
        filename='palm_detection_mediapipe_2023feb.onnx'),
    Model(
        name='MediaPipe Hand Landmarker (ONNX)',
        url='https://github.com/opencv/opencv_zoo/raw/56cef36ae45e5a6da7eba01a91631f6d7e955da1/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx',
        sha='48cfa3de98f30986ae2be6ed55e80d46e06713ab',
        filename='handpose_estimation_mediapipe_2023feb.onnx'),
    Model(
        name='MediaPipe Pose Landmarker (ONNX)',
        url='https://github.com/opencv/opencv_zoo/raw/1f19f821d68288feff2ef5c53993b33da74b1509/models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx',
        sha='9ecbfab8dec975ba02d8436a65cd69755238be20',
        filename='pose_estimation_mediapipe_2023mar.onnx'),
    Model(
        name='PaddlePaddle Human Segmentation (ONNX)',
        url='https://github.com/opencv/opencv_zoo/raw/2027dd2f5a8a5746b5d4964900a0465afc6d3a53/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx',
        sha='f0fec695ab7b716eeab4c58b125e98fc3826bb72',
        filename='human_segmentation_pphumanseg_2023mar.onnx'),
    Model(
        name='CRNN (ONNX)',
        url='https://github.com/opencv/opencv_zoo/raw/aab69020085e9b6390723b61f9789ec56b96b07e/models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx',
        sha='dc8c70a52c6880f11859bf074bcd294a45860821',
        filename='text_recognition_CRNN_EN_2021sep.onnx'),
    Model(
        name='RAFT', # See https://github.com/opencv/opencv_zoo/tree/main/models/optical_flow_estimation_raft#raft for source
        url='https://github.com/opencv/opencv_zoo/raw/281d232cd99cd920853106d853c440edd35eb442/models/optical_flow_estimation_raft/optical_flow_estimation_raft_2023aug.onnx',
        sha='8165e43c7bd37cc651f66509532acdb3c292832b',
        filename='onnx/models/optical_flow_estimation_raft_2023aug.onnx'),
    Model(
        name='vit_b_32',
        url=[
            'https://drive.google.com/u/0/uc?id=1UEeAyBs76XVkypk56ou7B8rBEIAlirkD&export=download', # See https://github.com/opencv/opencv_extra/pull/1128 to generate this model from torchvision
            'https://dl.opencv.org/models/vit/vit_b_32.onnx',
        ],
        sha='88144dca52cf3c6fee3aed8f8ca5c0b431e0afbd',
        filename='onnx/models/vit_b_32.onnx'),
    Model(
        name='object_tracking_vittrack',
        url=[
            'https://github.com/opencv/opencv_zoo/raw/fef72f8fa7c52eaf116d3df358d24e6e959ada0e/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx',
        ],
        sha='50008bb4f6a27b1aa940ad886b1bd1936ac4ed3e',
        filename='onnx/models/object_tracking_vittrack_2023sep.onnx'),

    # Original Intel Open Model Zoo models
    Model(
        name='age-gender-recognition-retail-0013-fp32 (xml)',
        url=[
            'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml',
            'https://dl.opencv.org/models/intel_open_model_zoo/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml'
        ],
        sha='6d0789605fa378af8bce0ec0f9723bdd356aaf62',
        filename='../intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml'),
    Model(
        name='age-gender-recognition-retail-0013-fp32 (weights)',
        url=[
            'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin',
            'https://dl.opencv.org/models/intel_open_model_zoo/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin'
        ],
        sha='4a31977cdb95bb153de3b949003a977d4ea4ed07',
        filename='../intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin'),
    Model(
        name='age-gender-recognition-retail-0013-fp16 (xml)',
        url=[
            'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml',
            'https://dl.opencv.org/models/intel_open_model_zoo/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml'
        ],
        sha='fa6872f82ee3ab9cbaca326d9191c1ce7b717ece',
        filename='../intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml'),
    Model(
        name='age-gender-recognition-retail-0013-fp16 (weights)',
        url=[
            'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin',
            'https://dl.opencv.org/models/intel_open_model_zoo/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin'
        ],
        sha='c6c0d0e57cdebece1b09794043caca0ca097532e',
        filename='../intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin'),
    Model(
        name='person-detection-retail-0013-fp32 (xml)',
        url=[
            'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP32/person-detection-retail-0013.xml',
            'https://dl.opencv.org/models/intel_open_model_zoo/person-detection-retail-0013/FP32/person-detection-retail-0013.xml'
        ],
        sha='df95657b05e5affc1c89165bd8c29b0e15dcdea9',
        filename='../intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml'),
    Model(
        name='person-detection-retail-0013-fp32 (weights)',
        url=[
            'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-detection-retail-0013/FP32/person-detection-retail-0013.bin',
            'https://dl.opencv.org/models/intel_open_model_zoo/person-detection-retail-0013/FP32/person-detection-retail-0013.bin'
        ],
        sha='682e59855466f88eb0cab9d40ca16e9fd6303bea',
        filename='../intel/person-detection-retail-0013/FP32/person-detection-retail-0013.bin'),
    Model(
        name='MediaPipe Blendshape V2 (TFLite)',
        url='https://storage.googleapis.com/mediapipe-assets/face_blendshapes.tflite?generation=1677787708051579',
        sha='eaf27df74abb6e112f3edbd7b06eb3d464fd02cc',
        filename='tflite/face_blendshapes.tflite'),
]

# Note: models will be downloaded to current working directory
#       expected working directory is <testdata>/dnn
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Download test models for OpenCV library")
    parser.add_argument("-d", "--dst", "--destination", help="Destination folder", default=Path.cwd())
    parser.add_argument("-l", "--list", action="store_true", help="List models")
    parser.add_argument("-r", "--ref", "--reference", help="Reference directory containing pre-downloaded models (read-only cache)")
    parser.add_argument("--cleanup", action="store_true", help="Remove archives after download")
    parser.add_argument("model", nargs='*', help="Model name to download (substring, case-insensitive)")
    args = parser.parse_args()
    ref = Path(args.ref).absolute() if args.ref else None

    # Apply filters
    filtered = []
    if args.model and len(args.model) > 0:
        for m in models:
            matches = [pat.lower() in m.name.lower() for pat in args.model]
            if matches.count(True) > 0:
                filtered.append(m)
        if len(filtered) == 0:
            print("No models match the filter")
            exit(14)
        else:
            print("Filtered: {} models".format(len(filtered)))
    else:
        filtered = models

    # List models
    if args.list:
        for mdl in filtered:
            print(mdl.name)
        exit()

    # Destination directory
    dest = Path(args.dst)
    if not dest.is_dir():
        print('  creating directory: {}'.format(dest))
        dest.mkdir(parents=True, exist_ok=True)
    os.chdir(dest)

    # Actual download
    proc = Processor(reference=ref)
    results = dict()
    for mdl in filtered:
        t = time.time()
        results[mdl] = proc.get(mdl)
        print("* {} ({:.2f} sec)".format("OK" if results[mdl] else "FAIL", time.time() - t))

    # Result handling
    for (mdl, res) in results.items():
        if args.cleanup and res and mdl.is_archive():
            proc.cleanup(mdl.filename)
        if not res:
            print("FAILED: {} - {}".format(mdl.name, mdl.filename))
    if list(results.values()).count(False) > 0:
        exit(15)
    else:
        print("SUCCESS")
