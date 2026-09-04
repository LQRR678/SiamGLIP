# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.atss_assigner import ATSSAssigner


@TASK_UTILS.register_module()
class CustomATSSAssigner(ATSSAssigner):
    """Custom ATSS Assigner that inherits from ATSSAssigner and handles Siamese compatibility issues."""

    def __init__(self,
                 topk: int = 9,
                 alpha: Optional[float] = None,
                 iou_calculator: dict = dict(type='BboxOverlaps2D'),
                 ignore_iof_thr: float = -1):
        super().__init__(
            topk=topk,
            alpha=alpha,
            iou_calculator=iou_calculator,
            ignore_iof_thr=ignore_iof_thr
        )

    def assign(self,
               pred_instances: InstanceData,
               num_level_priors: List[int],
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None) -> AssignResult:
        """Assign gt to priors with Siamese compatibility handling.

        This method handles the compatibility issues specific to Siamese networks:
        1. Empty bboxes/priors handling
        2. Tensor type compatibility
        3. Safe attribute access

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            num_level_priors (List[int]): Number of bboxes in each level
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        # Safe attribute access for Siamese compatibility
        gt_bboxes = getattr(gt_instances, 'bboxes', None)
        priors = getattr(pred_instances, 'priors', None)
        gt_labels = getattr(gt_instances, 'labels', None)
        
        # Handle empty cases that might occur in Siamese networks
        if gt_bboxes is None or gt_bboxes.size(0) == 0:
            # No ground truth, assign everything to background
            if priors is not None and priors.size(0) > 0:
                num_priors = priors.size(0)
                assigned_gt_inds = priors.new_full((num_priors, ), 0, dtype=torch.long)
                assigned_labels = priors.new_full((num_priors, ), -1, dtype=torch.long)
                max_overlaps = priors.new_zeros((num_priors, ))
            else:
                assigned_gt_inds = torch.tensor([], dtype=torch.long)
                assigned_labels = torch.tensor([], dtype=torch.long)
                max_overlaps = torch.tensor([], dtype=torch.float)
            return AssignResult(
                num_gts=0,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

        if priors is None or priors.size(0) == 0:
            # No priors, return empty assignment
            num_gts = gt_bboxes.size(0)
            assigned_gt_inds = gt_bboxes.new_full((0, ), -1, dtype=torch.long)
            assigned_labels = gt_bboxes.new_full((0, ), -1, dtype=torch.long)
            max_overlaps = gt_bboxes.new_zeros((0, ))
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

        # Ensure gt_labels exists for Siamese compatibility
        if gt_labels is None:
            # Create default labels if not present
            gt_labels = torch.zeros(gt_bboxes.size(0), dtype=torch.long, device=gt_bboxes.device)
            # Update gt_instances to include labels
            gt_instances.labels = gt_labels

        # Handle ignore instances safely for Siamese compatibility
        if gt_instances_ignore is not None:
            # Ensure gt_instances_ignore has bboxes attribute
            if not hasattr(gt_instances_ignore, 'bboxes'):
                # Create a safe ignore instance with empty bboxes
                from mmengine.structures import InstanceData
                safe_ignore = InstanceData()
                safe_ignore.bboxes = torch.empty((0, 4), dtype=gt_bboxes.dtype, device=gt_bboxes.device)
                gt_instances_ignore = safe_ignore
            else:
                gt_bboxes_ignore = getattr(gt_instances_ignore, 'bboxes', None)
                if gt_bboxes_ignore is None or gt_bboxes_ignore.size(0) == 0:
                    # Create empty ignore instance
                    from mmengine.structures import InstanceData
                    safe_ignore = InstanceData()
                    safe_ignore.bboxes = torch.empty((0, 4), dtype=gt_bboxes.dtype, device=gt_bboxes.device)
                    gt_instances_ignore = safe_ignore
        else:
            gt_bboxes_ignore = None

        # Call parent class assign method with safe parameters
        try:
            return super().assign(
                pred_instances=pred_instances,
                num_level_priors=num_level_priors,
                gt_instances=gt_instances,
                gt_instances_ignore=gt_instances_ignore
            )
        except Exception as e:
            # Fallback to simple assignment if ATSS fails
            print(f"ATSS assignment failed, using fallback: {e}")
            return self._fallback_assign(
                pred_instances, num_level_priors, gt_instances, gt_instances_ignore
            )

    def _fallback_assign(self,
                        pred_instances: InstanceData,
                        num_level_priors: List[int],
                        gt_instances: InstanceData,
                        gt_instances_ignore: Optional[InstanceData] = None) -> AssignResult:
        """Fallback assignment method for Siamese compatibility."""
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels

        # Calculate IoU
        overlaps = self.iou_calculator(priors, gt_bboxes)
        
        # Simple assignment based on IoU
        num_priors = priors.size(0)
        num_gts = gt_bboxes.size(0)
        
        # Assign each prior to the gt with highest IoU
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        assigned_gt_inds = argmax_overlaps + 1  # 1-based indexing
        
        # Set negative samples (IoU < 0.5)
        assigned_gt_inds[max_overlaps < 0.5] = 0
        
        # Set positive samples (IoU >= 0.5)
        assigned_labels = assigned_gt_inds.new_full((num_priors, ), -1)
        pos_inds = assigned_gt_inds > 0
        if pos_inds.any():
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)