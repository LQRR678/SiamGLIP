import mmengine
import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from mmcv.transforms.utils import cache_randomness
import numpy as np
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.image import imresize
from .siamese_box_type import siamese_autocast_box_type

@TRANSFORMS.register_module()
class SiameseRandomChoiceResize(BaseTransform):
    """Resize images & bbox & mask from a list of multiple scales.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. Resize scale will be randomly
    selected from ``scales``.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if `scale` is a list of tuple, the target scale is sampled from the list
      uniformally.
    - if `scale` is a tuple, the target scale will be set to the tuple.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Added Keys:

    - scale
    - scale_factor
    - scale_idx
    - keep_ratio


    Args:
        scales (Union[list, Tuple]): Images scales for resizing.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        scales: Sequence[Union[int, Tuple]],
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:
        super().__init__()
        if isinstance(scales, list):
            self.scales = scales
        else:
            self.scales = [scales]
        assert mmengine.is_seq_of(self.scales, (tuple, int))

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Resize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    @cache_randomness
    def _random_select(self) -> Tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(self.scales))
        scale = self.scales[scale_idx]
        return scale, scale_idx

    def transform(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        target_scale, scale_idx = self._random_select()
        self.resize.scale = target_scale
        results = self.resize(results)
        results['scale_idx'] = scale_idx
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scales={self.scales}'
        repr_str += f', resize_cfg={self.resize_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class SiameseResize(MMCV_Resize):
    """Resize images & bbox & seg.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

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

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                results['gt_masks'] = results['gt_masks'].rescale(
                    results['scale'])
            else:
                results['gt_masks'] = results['gt_masks'].resize(
                    results['img_shape'])

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('T_gt_bboxes', None) is not None:
            results['T_gt_bboxes'].rescale_(results['T_scale_factor'])
            if self.clip_object_border:
                results['T_gt_bboxes'].clip_(results['T_img_shape'])
        if results.get('S_gt_bboxes', None) is not None:
            results['S_gt_bboxes'].rescale_(results['S_scale_factor'])
            if self.clip_object_border:
                results['S_gt_bboxes'].clip_(results['S_img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['T_scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('T_homography_matrix', None) is None:
            results['T_homography_matrix'] = homography_matrix
        else:
            results['T_homography_matrix'] = homography_matrix @ results[
                'T_homography_matrix']
        
        w_scale, h_scale = results['S_scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('S_homography_matrix', None) is None:
            results['S_homography_matrix'] = homography_matrix
        else:
            results['S_homography_matrix'] = homography_matrix @ results[
                'S_homography_matrix']

    @siamese_autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['T_scale'] = self.scale
            results['S_scale'] = self.scale
        else:
            T_img_shape = results['T_img'].shape[:2]
            results['T_scale'] = _scale_size(T_img_shape[::-1], self.scale_factor)
            S_img_shape = results['S_img'].shape[:2]
            results['S_scale'] = _scale_size(S_img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


def _fixed_scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    # don't need o.5 offset
    return int(w * float(scale[0])), int(h * float(scale[1]))


def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')
    # only change this
    new_size = _fixed_scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: np.ndarray,
    scale: Union[float, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    backend: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


@TRANSFORMS.register_module()
class SiameseFixScaleResize(SiameseResize):
    """Compared to Resize, FixScaleResize fixes the scaling issue when
    `keep_ratio=true`."""

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if results.get('T_img', None) is not None:
            if self.keep_ratio:
                T_img, T_scale_factor = imrescale(
                    results['T_img'],
                    results['T_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
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
                S_img, S_scale_factor = imrescale(
                    results['S_img'],
                    results['S_scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
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
