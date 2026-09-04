# Copyright (c) SiamGLIP Authors. All rights reserved.

def compute_iou_with_center_anchor(cx: float, cy: float, bw: float, bh: float) -> float:
  
    # ---------- 目标框（裁剪到 [0, 1]）----------
    b_x1 = max(0.0, cx - bw / 2.0)
    b_y1 = max(0.0, cy - bh / 2.0)
    b_x2 = min(1.0, cx + bw / 2.0)
    b_y2 = min(1.0, cy + bh / 2.0)

    # ---------- 中心 anchor（同尺度，中心对齐到 0.5, 0.5）----------
    a_x1 = max(0.0, 0.5 - bw / 2.0)
    a_y1 = max(0.0, 0.5 - bh / 2.0)
    a_x2 = min(1.0, 0.5 + bw / 2.0)
    a_y2 = min(1.0, 0.5 + bh / 2.0)

    # ---------- 交集 ----------
    inter_x1 = max(b_x1, a_x1)
    inter_y1 = max(b_y1, a_y1)
    inter_x2 = min(b_x2, a_x2)
    inter_y2 = min(b_y2, a_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # ---------- 并集 ----------
    bbox_area = max(0.0, b_x2 - b_x1) * max(0.0, b_y2 - b_y1)
    anchor_area = max(0.0, a_x2 - a_x1) * max(0.0, a_y2 - a_y1)
    union_area = bbox_area + anchor_area - inter_area

    if union_area < 1e-8:
        return 0.0

    return inter_area / union_area


def is_center_by_iou(
    cx: float, cy: float, bw: float, bh: float, iou_thr: float = 0.5
) -> bool:
    
    iou = compute_iou_with_center_anchor(cx, cy, bw, bh)
    return iou >= iou_thr
