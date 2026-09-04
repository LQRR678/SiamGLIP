# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import numpy as np
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.base_boxes import BaseBoxes
from torch import Tensor

BoxType = Union[np.ndarray, Tensor, BaseBoxes]

box_types: dict = {}
_box_type_to_name: dict = {}
box_converters: dict = {}


def siamese_autocast_box_type(dst_box_type="hbox") -> Callable:
    """A decorator which automatically casts results['gt_bboxes'] to the
    destination box type.

    It commenly used in mmdet.datasets.transforms to make the transforms up-
    compatible with the np.ndarray type of results['gt_bboxes'].

    The speed of processing of np.ndarray and BaseBoxes data are the same:

    - np.ndarray: 0.0509 img/s
    - BaseBoxes: 0.0551 img/s

    Args:
        dst_box_type (str): Destination box type.
    """
    _, box_type_cls = get_box_type(dst_box_type)

    def decorator(func: Callable) -> Callable:
        def wrapper(self, results: dict, *args, **kwargs) -> dict:
            if "T_gt_bboxes" not in results or isinstance(
                results["T_gt_bboxes"], BaseBoxes
            ):
                return func(self, results)
            elif isinstance(results["T_gt_bboxes"], np.ndarray):
                results["T_gt_bboxes"] = box_type_cls(
                    results["T_gt_bboxes"], clone=False
                )
                results["S_gt_bboxes"] = box_type_cls(
                    results["S_gt_bboxes"], clone=False
                )
                if "mix_results" in results:
                    for res in results["mix_results"]:
                        if isinstance(res["T_gt_bboxes"], np.ndarray):
                            res["T_gt_bboxes"] = box_type_cls(
                                res["T_gt_bboxes"], clone=False
                            )
                        if isinstance(res["S_gt_bboxes"], np.ndarray):
                            res["S_gt_bboxes"] = box_type_cls(
                                res["S_gt_bboxes"], clone=False
                            )

                _results = func(self, results, *args, **kwargs)

                # In some cases, the function will process gt_bboxes in-place
                # Simultaneously convert inputting and outputting gt_bboxes
                # back to np.ndarray
                if isinstance(_results, dict) and "T_gt_bboxes" in _results:
                    if isinstance(_results["T_gt_bboxes"], BaseBoxes):
                        _results["T_gt_bboxes"] = _results["T_gt_bboxes"].numpy()
                        _results["S_gt_bboxes"] = _results["S_gt_bboxes"].numpy()
                if isinstance(results["T_gt_bboxes"], BaseBoxes):
                    results["T_gt_bboxes"] = results["T_gt_bboxes"].numpy()
                    results["S_gt_bboxes"] = results["S_gt_bboxes"].numpy()
                return _results
            else:
                raise TypeError(
                    "auto_box_type requires results['T_gt_bboxes'] to "
                    "be BaseBoxes or np.ndarray, but got "
                    f"{type(results['T_gt_bboxes'])}"
                )

        return wrapper

    return decorator
