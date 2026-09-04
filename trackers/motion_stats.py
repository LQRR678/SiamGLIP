from typing import Optional

import torch
from torch import Tensor


def _bbox_center_xyxy(box: Tensor) -> Tensor:
    """Return (x, y) center of a bbox in xyxy format."""
    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dim=-1)


def compute_motion_vec_from_boxes(
    boxes: Tensor,
    motion_history_max_len: int,
    frame_ids: Optional[Tensor] = None,
    dt: float = 1.0,
    stab_sigma: float = 1.0,
    eps: float = 1e-6,
    fmt: str = "xyxy",
    img_shape: Optional[tuple] = None,
    include_geom: bool = True,
    return_sequence: bool = True, 
    *,
    max_len: Optional[int] = None,
) -> Tensor:
    """
    Compute motion_vec = [cx, cy, w, h], normalized per-axis by (W, H).
    """
    if max_len is not None:
        motion_history_max_len = max_len

    out_dim = 4
    device = boxes.device if isinstance(boxes, torch.Tensor) else None

    if boxes.numel() == 0:
        if return_sequence:
            return torch.zeros(1, out_dim, device=device)
        return torch.zeros(out_dim, device=device)

    # Accept (4,), (K, 4) or anything divisible by 4
    if boxes.dim() == 1:
        if boxes.numel() != 4:
            if return_sequence:
                return torch.zeros(1, out_dim, device=boxes.device)
            return torch.zeros(out_dim, device=boxes.device)
        boxes = boxes.view(1, 4)
    elif boxes.dim() != 2:
        if boxes.numel() % 4 != 0:
            if return_sequence:
                return torch.zeros(1, out_dim, device=boxes.device)
            return torch.zeros(out_dim, device=boxes.device)
        boxes = boxes.view(-1, 4)

    if boxes.shape[-1] != 4:
        if return_sequence:
            return torch.zeros(1, out_dim, device=boxes.device)
        return torch.zeros(out_dim, device=boxes.device)

    if fmt not in ("xyxy", "xywh"):
        raise ValueError(f"Unsupported box format: {fmt}")
    if fmt == "xywh":
        x1y1 = boxes[:, :2]
        wh = boxes[:, 2:]
        boxes = torch.cat([x1y1, x1y1 + wh], dim=-1)

    if motion_history_max_len is not None and boxes.shape[0] > motion_history_max_len:
        boxes = boxes[-motion_history_max_len:]

    # 计算每帧的 [cx, cy, w, h]
    centers = _bbox_center_xyxy(boxes)  # (K, 2)
    widths = boxes[:, 2] - boxes[:, 0]  # (K,)
    heights = boxes[:, 3] - boxes[:, 1]  # (K,)

    # 归一化系数
    w_norm = h_norm = 1.0
    if img_shape is not None and len(img_shape) >= 2:
        h_img, w_img = img_shape[:2]
        w_norm = max(float(w_img), eps)
        h_norm = max(float(h_img), eps)

    # 归一化后的时序数据: (K, 4)
    cx_seq = centers[:, 0] / w_norm  # (K,)
    cy_seq = centers[:, 1] / h_norm  # (K,)
    bw_seq = widths / w_norm  # (K,)
    bh_seq = heights / h_norm  # (K,)

    motion_sequence = torch.stack([cx_seq, cy_seq, bw_seq, bh_seq], dim=-1)  # (K, 4)

    return motion_sequence
