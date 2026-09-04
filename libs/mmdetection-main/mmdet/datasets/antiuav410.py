# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class AntiUAV410Dataset(CocoDataset):
    """Dataset for AntiUAV410Dataset."""

    METAINFO = {
        'classes': ('uav'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 192, 64)]
    }
