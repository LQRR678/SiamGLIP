import numpy as np
import torch
from typing import Optional, Sequence, Union
from mmcv.transforms import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes


@TRANSFORMS.register_module()
class SiameseBuildMotionHistory(BaseTransform):
    """Build motion_history from GT bboxes for current sample.

    Expects input results dict to have history of GT boxes (xyxy or xywh).
    It will write:
      - results['motion_history']: ndarray/Tensor (K,4) in xyxy order
      - results['motion_history_format']: 'xyxy' (for downstream computation)

    You need to ensure results['motion_history_gt'] is prepared upstream,
    e.g., by dataset providing past K GT boxes. This transform only does
    format conversion and renaming for downstream pack.
    """

    def __init__(self,
                 motion_history_max_len: int,
                 history_key: str = 'motion_history_gt',
                 fmt: str = 'xyxy',
                 glip_format: bool = True,
                 *,
                 max_len: Optional[int] = None):
        """
        Args:
            history_key: Key in results dict containing history boxes.
            fmt: Input box format, 'xyxy' or 'xywh'.
            motion_history_max_len: Maximum number of history frames to keep.
            glip_format: If True, subtract 1 from x2, y2 to match GLIP's
                closed-interval bbox format (consistent with SiameseGTBoxSubOne_GLIP).
            max_len: Deprecated alias of motion_history_max_len.
        """
        if max_len is not None:
            motion_history_max_len = max_len
        self.history_key = history_key
        self.fmt = fmt
        self.motion_history_max_len = motion_history_max_len
        self.glip_format = glip_format

    @staticmethod
    def _apply_homography(boxes: torch.Tensor,
                          homography: torch.Tensor) -> torch.Tensor:
        """仿射/缩放/翻转."""
        if boxes.numel() == 0:
            return boxes
        # (N,4) -> (N,4,2) corners
        hboxes = HorizontalBoxes(boxes, clone=False)
        corners = hboxes.hbox2corner(hboxes.tensor)  # (N,4,2)
        ones = corners.new_ones(*corners.shape[:-1], 1)
        corners_h = torch.cat([corners, ones], dim=-1)  # (N,4,3)
        # 展平 batch 应用变换
        flat = corners_h.view(-1, 3).t()  # (3, N*4)
        warped = homography @ flat  # (3, N*4)
        warped = warped[:2] / warped[2:].clamp(min=1e-6)
        warped = warped.t().view(*corners.shape[:-1], 2)
        # corners -> xyxy
        warped_boxes = HorizontalBoxes.corner2hbox(warped)
        return warped_boxes

    def transform(self, results: dict) -> dict:
        if self.history_key not in results:
            return results
        history = results[self.history_key]
        if history is None:
            return results

        frame_ids = results.get(f'{self.history_key}_frame_ids', None)

        if isinstance(history, torch.Tensor):
            boxes = history.detach().cpu().float()
        else:
            boxes = torch.as_tensor(history, dtype=torch.float32)

        if self.motion_history_max_len is not None and boxes.shape[0] > self.motion_history_max_len:
            boxes = boxes[-self.motion_history_max_len:]
            if frame_ids is not None:
                frame_ids = frame_ids[-self.motion_history_max_len:]

        if self.fmt == 'xywh':
            x1y1 = boxes[:, :2]
            wh = boxes[:, 2:]
            boxes = torch.cat([x1y1, x1y1 + wh], dim=-1)
        elif self.fmt != 'xyxy':
            raise ValueError(f'Unsupported bbox format: {self.fmt}')

        if self.glip_format and boxes.numel() > 0:
            boxes[:, 2:] -= 1

        homographies = []
        if 'S_homography_matrix' in results:
            homographies.append(torch.as_tensor(
                results['S_homography_matrix'], dtype=torch.float32))
        if 'homography_matrix' in results:
            homographies.append(torch.as_tensor(
                results['homography_matrix'], dtype=torch.float32))
        if homographies:
            H = homographies[0]
            for h in homographies[1:]:
                H = h @ H
            boxes = self._apply_homography(boxes, H)
            results['motion_history_is_resized'] = True

        results['motion_history'] = boxes.numpy()
        if frame_ids is not None:
            results['motion_history_frame_ids'] = np.asarray(frame_ids)
        results['motion_history_format'] = 'xyxy'
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(history_key={self.history_key}, '
                f'fmt={self.fmt}, motion_history_max_len={self.motion_history_max_len}, glip_format={self.glip_format})')
