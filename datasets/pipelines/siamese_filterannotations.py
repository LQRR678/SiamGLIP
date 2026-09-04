from typing import Optional, Tuple, Union
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from .siamese_box_type import siamese_autocast_box_type


@TRANSFORMS.register_module()
class SiameseFilterAnnotations(BaseTransform):
    """Filter invalid annotations.

    Required Keys:

    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground truth
            boxes. Default: (1., 1.)
        min_gt_mask_area (int): Minimum foreground area of ground truth masks.
            Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: True
        by_mask (bool): Filter instances with masks not meeting
            min_gt_mask_area threshold. Default: False
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_mask_area: int = 1,
                 by_box: bool = True,
                 by_mask: bool = False,
                 keep_empty: bool = True) -> None:
        # TODO: add more filter options
        assert by_box or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_box
        self.by_mask = by_mask
        self.keep_empty = keep_empty

    @siamese_autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'T_gt_bboxes' in results
        T_gt_bboxes = results['T_gt_bboxes']
        if T_gt_bboxes.shape[0] == 0:
            if self.keep_empty:
                return None
            return results
        
        assert 'S_gt_bboxes' in results
        S_gt_bboxes = results['S_gt_bboxes']
        if S_gt_bboxes.shape[0] == 0:
            if self.keep_empty:
                return None
            return results

        tests = []
        if self.by_box:
            tests.append(
                ((T_gt_bboxes.widths > self.min_gt_bbox_wh[0]) &
                 (T_gt_bboxes.heights > self.min_gt_bbox_wh[1])&
                 (S_gt_bboxes.widths > self.min_gt_bbox_wh[0]) &
                 (S_gt_bboxes.heights > self.min_gt_bbox_wh[1])).numpy())
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None
        
        keys = ('T_gt_bboxes','S_gt_bboxes', 'gt_bboxes_labels', 'gt_masks', 'gt_ignore_flags')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]
                
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(min_gt_bbox_wh={self.min_gt_bbox_wh}, ' \
               f'keep_empty={self.keep_empty})'
