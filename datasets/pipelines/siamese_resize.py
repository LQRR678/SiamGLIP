import mmengine
import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from mmcv.transforms.utils import cache_randomness
import numpy as np

@TRANSFORMS.register_module()
class SiameseResize(BaseTransform):
    """Resize images & bbox & seg & keypoints.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('T_img', None) is not None:
            if self.keep_ratio:
                T_img, T_scale_factor = mmcv.imrescale(
                    results['T_img'],
                    results['T_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = T_img.shape[:2]
                h, w = results['T_img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                T_img, w_scale, h_scale = mmcv.imresize(
                    results['T_img'],
                    results['T_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['T_img'] = T_img
            results['T_img_shape'] = T_img.shape[:2]
            results['T_scale_factor'] = (w_scale, h_scale)
            results['T_keep_ratio'] = self.keep_ratio
        
        if results.get('S_img', None) is not None:
            if self.keep_ratio:
                S_img, S_scale_factor = mmcv.imrescale(
                    results['S_img'],
                    results['S_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = S_img.shape[:2]
                h, w = results['S_img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                S_img, w_scale, h_scale = mmcv.imresize(
                    results['S_img'],
                    results['S_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['S_img'] = S_img
            results['S_img_shape'] = S_img.shape[:2]
            results['S_scale_factor'] = (w_scale, h_scale)
            results['S_keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('T_gt_bboxes', None) is not None:
            T_bboxes = results['T_gt_bboxes'] * np.tile(
                np.array(results['T_scale_factor']), 2)
            if self.clip_object_border:
                T_bboxes[:, 0::2] = np.clip(T_bboxes[:, 0::2], 0,
                                          results['T_img_shape'][1])
                T_bboxes[:, 1::2] = np.clip(T_bboxes[:, 1::2], 0,
                                          results['T_img_shape'][0])
            results['T_gt_bboxes'] = T_bboxes
        
        if results.get('S_gt_bboxes', None) is not None:
            S_bboxes = results['S_gt_bboxes'] * np.tile(
                np.array(results['S_scale_factor']), 2)
            if self.clip_object_border:
                S_bboxes[:, 0::2] = np.clip(S_bboxes[:, 0::2], 0,
                                          results['S_img_shape'][1])
                S_bboxes[:, 1::2] = np.clip(S_bboxes[:, 1::2], 0,
                                          results['S_img_shape'][0])
            results['S_gt_bboxes'] = S_bboxes

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['T_scale'] = self.scale
            results['S_scale'] = self.scale
        else:
            T_img_shape = results['T_img'].shape[:2]
            results['T_scale'] = _scale_size(T_img_shape[::-1],
                                           self.scale_factor)  # type: ignore
            S_img_shape = results['S_img'].shape[:2]
            results['S_scale'] = _scale_size(S_img_shape[::-1],
                                           self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
