'''Canonical PT Caffe VGG16 asset specification.'''

CHECKPOINT_PATH = 'work_dirs/pretrained/vgg16_caffe.pth'
DOWNLOAD_URL = (
    'https://zenodo.org/records/4515252/files/vgg16_caffe.pth?download=1'
)
SHA256 = (
    '736b4bd0b787438253ea1926f9a02730b2eedbf0e48df243457d17133fe8850e'
)
MD5 = '433ad40ddbd662d6448e13a6cef812f2'
SIZE_BYTES = 553433685

__all__ = [
    'CHECKPOINT_PATH',
    'DOWNLOAD_URL',
    'MD5',
    'SHA256',
    'SIZE_BYTES',
]
