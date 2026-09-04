# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes
from .siamese_box_type import siamese_autocast_box_type
from mmdet.datasets.transforms.transforms import RandomFlip


@TRANSFORMS.register_module()
class SiameseGTBoxSubOne_GLIP(BaseTransform):
    """Subtract 1 from the x2 and y2 coordinates of the gt_bboxes."""

    def transform(self, results: dict) -> dict:
        if 'T_gt_bboxes' in results:
            T_gt_bboxes = results['T_gt_bboxes']
            if isinstance(T_gt_bboxes, np.ndarray):
                T_gt_bboxes[:, 2:] -= 1
                results['T_gt_bboxes'] = T_gt_bboxes
            elif isinstance(T_gt_bboxes, HorizontalBoxes):
                T_gt_bboxes = results['T_gt_bboxes'].tensor
                T_gt_bboxes[:, 2:] -= 1
                results['T_gt_bboxes'] = HorizontalBoxes(T_gt_bboxes)
            else:
                raise NotImplementedError

        if 'S_gt_bboxes' in results:
            S_gt_bboxes = results['S_gt_bboxes']
            if isinstance(S_gt_bboxes, np.ndarray):
                S_gt_bboxes[:, 2:] -= 1
                results['S_gt_bboxes'] = S_gt_bboxes
            elif isinstance(S_gt_bboxes, HorizontalBoxes):
                S_gt_bboxes = results['S_gt_bboxes'].tensor
                S_gt_bboxes[:, 2:] -= 1
                results['S_gt_bboxes'] = HorizontalBoxes(S_gt_bboxes)
            else:
                raise NotImplementedError
        return results


@TRANSFORMS.register_module()
class SiameseRandomFlip_GLIP(RandomFlip):
    """Flip the image & bboxes & masks & segs horizontally or vertically.

    When using horizontal flipping, the corresponding bbox x-coordinate needs
    to be additionally subtracted by one.
    """
    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results['flip_direction']
        h, w = results['T_img'].shape[:2]

        if cur_dir == 'horizontal':
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'vertical':
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'diagonal':
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    @siamese_autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['T_img'] = mmcv.imflip(
            results['T_img'], direction=results['flip_direction'])

        T_img_shape = results['T_img'].shape[:2]

        # flip bboxes
        if results.get('T_gt_bboxes', None) is not None:
            results['T_gt_bboxes'].flip_(T_img_shape, results['flip_direction'])
            # Only change this line
            if results['flip_direction'] == 'horizontal':
                results['T_gt_bboxes'].translate_([-1, 0])

        # flip image
        results['S_img'] = mmcv.imflip(
            results['S_img'], direction=results['flip_direction'])

        S_img_shape = results['S_img'].shape[:2]

        # flip bboxes
        if results.get('S_gt_bboxes', None) is not None:
            results['S_gt_bboxes'].flip_(T_img_shape, results['flip_direction'])
            # Only change this line
            if results['flip_direction'] == 'horizontal':
                results['S_gt_bboxes'].translate_([-1, 0])


        # TODO: check it
        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)
