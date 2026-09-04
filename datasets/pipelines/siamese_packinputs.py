# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from trackers.motion_stats import compute_motion_vec_from_boxes


@TRANSFORMS.register_module()
class SiamesePackDetInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'T_gt_bboxes': 'T_bboxes',
        'S_gt_bboxes': 'S_bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 motion_history_max_len: int,
                 meta_keys=('img_id', 'T_img_path', 'S_img_path', 'T_ori_shape', 'S_ori_shape', 'T_img_shape', 'S_img_shape',
                             'T_scale_factor','S_scale_factor', 'flip', 'flip_direction'),
                 motion_noise_std_range=None,
                 motion_drop_prob: float = 0.0,
                 motion_clip_max=None,
                 *,
                 max_len: Optional[int] = None):
        if max_len is not None:
            motion_history_max_len = max_len
        self.meta_keys = meta_keys
        self.motion_history_max_len = motion_history_max_len
        self.motion_noise_std_range = motion_noise_std_range
        self.motion_drop_prob = motion_drop_prob
        self.motion_clip_max = motion_clip_max
        # 通过 results.get('training', False) 判断是否在训练模式

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'T_img' in results:
            T_img = results['T_img']
            if len(T_img.shape) < 3:
                T_img = np.expand_dims(T_img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not T_img.flags.c_contiguous:
                T_img = np.ascontiguousarray(T_img.transpose(2, 0, 1))
                T_img = to_tensor(T_img)
            else:
                T_img = to_tensor(T_img).permute(2, 0, 1).contiguous()

            packed_results['T_inputs'] = T_img

        if 'S_img' in results:
            S_img = results['S_img']
            if len(S_img.shape) < 3:
                S_img = np.expand_dims(S_img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not S_img.flags.c_contiguous:
                S_img = np.ascontiguousarray(S_img.transpose(2, 0, 1))
                S_img = to_tensor(S_img)
            else:
                S_img = to_tensor(S_img).permute(2, 0, 1).contiguous()

            packed_results['S_inputs'] = S_img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            elif isinstance(results[key], BaseBoxes):
                # 对于 BaseBoxes 对象，需要转换为 tensor
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx].tensor
                    ignore_instance_data[self.mapping_table[key]] = results[key][ignore_idx].tensor
                else:
                    instance_data[self.mapping_table[key]] = results[key].tensor
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        # 添加标准的 bboxes 字段，使用 T_bboxes 作为默认值
        if hasattr(instance_data, 'T_bboxes') and instance_data.T_bboxes.numel() > 0:
            instance_data.bboxes = instance_data.T_bboxes
        elif hasattr(instance_data, 'S_bboxes') and instance_data.S_bboxes.numel() > 0:
            instance_data.bboxes = instance_data.S_bboxes
            
        if 'gt_texts' in results:
            instance_data.text = results['gt_texts']

        if hasattr(ignore_instance_data, 'T_bboxes') and ignore_instance_data.T_bboxes.numel() > 0:
            ignore_instance_data.bboxes = ignore_instance_data.T_bboxes
        elif hasattr(ignore_instance_data, 'S_bboxes') and ignore_instance_data.S_bboxes.numel() > 0:
            ignore_instance_data.bboxes = ignore_instance_data.S_bboxes
            
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results and key not in ('text', 'test'):
                img_meta[key] = results[key]
        if 'motion_size_diag' in results:
            img_meta['motion_size_diag'] = results['motion_size_diag']
        # motion history (xyxy) -> motion_vec metainfo
        motion_hist = results.get('motion_history', None)
        if motion_hist is not None:
            # expect list/ndarray/torch.Tensor of shape (K,4) in xyxy
            if isinstance(motion_hist, torch.Tensor):
                boxes = motion_hist.float()
            else:
                boxes = torch.as_tensor(motion_hist, dtype=torch.float32)
            fmt = results.get('motion_history_format', 'xyxy')

            frame_ids = results.get('motion_history_frame_ids', None)
            if frame_ids is not None:
                frame_ids = torch.as_tensor(frame_ids, dtype=torch.float32)
                # 对齐长度
                if frame_ids.numel() != boxes.shape[0]:
                    min_len = min(frame_ids.numel(), boxes.shape[0])
                    frame_ids = frame_ids[-min_len:]
                    boxes = boxes[-min_len:]
            
            # 确定归一化使用的图像尺寸
            if results.get('motion_history_is_resized', False):
                norm_shape = results.get('S_img_shape', results.get('img_shape', None))
            else:
                norm_shape = results.get('S_ori_shape', results.get('ori_shape', None))

            motion_vec = compute_motion_vec_from_boxes(
                boxes,
                frame_ids=frame_ids,
                motion_history_max_len=self.motion_history_max_len,
                fmt=fmt,
                img_shape=norm_shape,
                include_geom=True)
            # 仅训练
            is_training = bool(results.get('training', False))
            if is_training:
                if self.motion_noise_std_range is not None:
                    low, high = self.motion_noise_std_range
                    std = float(torch.empty(1).uniform_(low, high))       # 在给定区间随机采样标准差 σ
                    motion_vec = motion_vec + torch.randn_like(motion_vec) * std  #添加高斯噪声
                if self.motion_drop_prob and float(torch.rand(1)) < self.motion_drop_prob:
                    if float(torch.rand(1)) < 0.5:
                        motion_vec = torch.zeros_like(motion_vec)
                    else:
                        mask = torch.rand_like(motion_vec) > 0.5
                        motion_vec = motion_vec * mask
                if self.motion_clip_max is not None:
                    mmax = float(self.motion_clip_max)
                    motion_vec = torch.clamp(motion_vec, -mmax, mmax)
            img_meta['motion_vec'] = motion_vec
    

        # 推理
        # 优先使用 T 分支的信息
        if 'T_img_shape' in results:
            img_meta['img_shape'] = results['T_img_shape']
        if 'T_ori_shape' in results:
            img_meta['ori_shape'] = results['T_ori_shape']
        if 'T_scale_factor' in results:
            img_meta['scale_factor'] = results['T_scale_factor']
        if 'T_img_path' in results:
            img_meta['img_path'] = results['T_img_path']
        data_sample.set_metainfo(img_meta)
        if 'text' in results:
            data_sample.text = results['text']
        if 'test' in results:
            data_sample.test = results['test']
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
