# Upstream framework and assets

The local `mmdet` package is vendored from the official MMDetection v3.3.0 tag.

- Repository: https://github.com/open-mmlab/mmdetection
- Tag: v3.3.0
- Commit: `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`
- License: Apache-2.0, copied to `mmdet/LICENSE`

ADAOD-specific Python files are not added inside that package. Project models,
transforms, and metrics register from `methods` through MMDetection's registry
mechanism.

## PT VGG16 pretrained asset

PT loads convolution tensors from a Caffe-converted VGG16 checkpoint. ADAOD
uses the preserved file below and pins its content before model construction.
The path, URL, SHA256, MD5, and byte size have one canonical specification in
`methods/common/mmdet/models/backbones/vgg16_caffe.py`; detector configuration,
asset preparation, and execution import that definition instead of duplicating
the constants.

- Preservation record: https://zenodo.org/records/4515252
- Download URL: https://zenodo.org/records/4515252/files/vgg16_caffe.pth?download=1
- Local cache: `work_dirs/pretrained/vgg16_caffe.pth`
- Size: 553,433,685 bytes
- MD5 published by the record: `433ad40ddbd662d6448e13a6cef812f2`
- SHA256 pinned by ADAOD: `736b4bd0b787438253ea1926f9a02730b2eedbf0e48df243457d17133fe8850e`

The download is HTTPS-only, streamed to a temporary sibling, SHA256-verified,
flushed, and atomically installed. Existing invalid files are rejected rather
than silently replaced. Offline execution verifies the cached file and never
accesses the network.
