'''VGG-style two-FC Faster R-CNN bbox head with MC-dropout layers.'''

from typing import Optional, Tuple

from torch import Tensor, nn

try:
    from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'VGGShared2FCBBoxHead requires the repository-local MMDetection 3.3 '
        'package. PyTorch-only components can be imported from '
        'methods.common.mmdet without importing this module.'
    ) from exc


class VGGShared2FCBBoxHead(Shared2FCBBoxHead):
    '''Shared fc6/fc7 bbox head with dropout after each ReLU.

    PT uses two newly initialized 1024-dimensional FC layers; its Caffe VGG16
    checkpoint initializes only the convolutional backbone.
    '''

    def __init__(self,
                 *args,
                 fc_out_channels: int = 1024,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        if dropout < 0 or dropout >= 1:
            raise ValueError('dropout must be in the interval [0, 1)')
        super().__init__(
            *args, fc_out_channels=fc_out_channels, **kwargs)
        if len(self.shared_fcs) != 2:
            raise RuntimeError('VGG bbox head requires exactly two shared FCs')
        self.dropout = float(dropout)
        self.shared_dropouts = nn.ModuleList(
            nn.Dropout(p=self.dropout) for _ in self.shared_fcs)

    def init_weights(self) -> None:
        '''Initialize fc6/fc7 with Detectron2's C2 Xavier fill.'''

        super().init_weights()
        for fc in self.shared_fcs:
            nn.init.kaiming_uniform_(fc.weight, a=1)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0)

    def forward(
            self, inputs: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        x = inputs
        for conv in self.shared_convs:
            x = conv(x)
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.flatten(1)
        for fc, dropout in zip(self.shared_fcs, self.shared_dropouts):
            x = dropout(self.relu(fc(x)))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred
