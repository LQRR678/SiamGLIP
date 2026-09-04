
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import glob
import json
import os
import os.path as osp
import warnings
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset
from PIL import Image


@DATASETS.register_module()
class AntiUAV410TrackingDataset(BaseDetDataset):
    """Dataset for COCO with Adaptive Text Prompts."""

    METAINFO = {
        'classes': ('uav'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 192, 64)],
 
        'adaptive_text_prompts': {
            'enable': True,
            # 'enable': False,
            'base_class': 'uav',
            'size_thresholds': {
                'tiny': (0, 10),
                'small': (10, 30),
                'normal': (30, 50),
                'large': (50, float('inf'))
            },
            'prompt_templates': {
                'tiny': [
                    'a tiny {}.',
                ],
                'small': [
                    'a small {}.',
                ],
                'normal': [
                    'a typical {}.',
                ],
                'large': [
                    'a large {}.',
                ]
            }
        }
    }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True 

    def __init__(self, motion_history_max_len: int, *args, **kwargs):
        self.motion_history_max_len = motion_history_max_len
        self._seq_motion_cache = {}
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)    #保存 COCO API 实例
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}   #类别 ID 到标签索引的映射
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        self._seq_motion_cache = {}

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]     #加载图像信息
            raw_img_info['img_id'] = img_id                     #补充图像 ID 到原始信息中

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)                       #收集所有标注 ID 到总列表

            parsed_data_info = self.parse_data_info({           #解析原始数据为模型可用格式
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
            # lazily build per-sequence motion cache from original IR_label.json
            try:
                search_rel = raw_img_info['file_name']['Search']
                seq_rel_dir = osp.dirname(search_rel)
                if seq_rel_dir not in self._seq_motion_cache:
                    img_prefix = self.data_prefix.get(
                        'img_path', self.data_prefix.get('img', ''))
                    seq_dir = osp.join(img_prefix, seq_rel_dir)
                    ir_label_path = osp.join(seq_dir, 'IR_label.json')
                    if osp.isfile(ir_label_path):
                        with open(ir_label_path, 'r') as f:
                            ir_data = json.load(f)
                        exist = ir_data.get('exist', [])
                        gt_rect = ir_data.get('gt_rect', [])
                        self._seq_motion_cache[seq_rel_dir] = dict(
                            exist=exist, gt_rect=gt_rect)
            except Exception:
                # motion cache is optional; fall back to zero motion if missing
                pass
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(  
                total_ann_ids                       #集合（set）的特性是元素不可重复，转换后会自动去除重复的 ID
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def _extract_sequence_id(self, template_rel_path: str) -> str:
        return osp.dirname(template_rel_path)

    def _list_sequence_frames(self, seq_rel_dir: str) -> List[str]:
        img_prefix = self.data_prefix.get('img_path',
                                          self.data_prefix.get('img', ''))
        seq_dir = osp.join(img_prefix, seq_rel_dir)
        if not osp.isdir(seq_dir):
            return []
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        frame_paths = []
        for pat in patterns:
            frame_paths.extend(glob.glob(osp.join(seq_dir, pat)))
        frame_paths.sort()
        return frame_paths

    def _calculate_target_size(self, bbox: List[float]) -> float:
        """
        计算目标尺寸（对角线长度）
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            float: 对角线长度
        """
        if len(bbox) != 4:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        diagonal = (width**2 + height**2)**0.5
        return diagonal
    
    def _get_size_category(self, size: float) -> str:
        """
        根据尺寸获取类别
        
        Args:
            size: 目标尺寸
            
        Returns:
            str: 尺寸类别
        """
        thresholds = self.metainfo.get('adaptive_text_prompts', {}).get('size_thresholds', {})
        
        for category, (min_size, max_size) in thresholds.items():
            if min_size <= size < max_size:
                return category
        return 'normal'  # 默认类别
    
    def _generate_adaptive_text_prompt(self, bbox: List[float]) -> str:
        """
        生成自适应文本提示
        
        Args:
            bbox: 目标边界框 [x1, y1, x2, y2]
            
        Returns:
            str: 自适应文本提示
        """
        # 计算目标尺寸
        size = self._calculate_target_size(bbox)
        category = self._get_size_category(size)
        
        # 获取提示模板
        templates = self.metainfo.get('adaptive_text_prompts', {}).get('prompt_templates', {})
        category_templates = templates.get(category, ['a {} .'])
        
        # 随机选择模板
        import random
        template = random.choice(category_templates)
        
        # 获取基础类别
        base_class = self.metainfo.get('adaptive_text_prompts', {}).get('base_class', 'uav')
        
        # 生成提示
        prompt = template.format(base_class)
        return prompt

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # 使用标准的img_path键名，与mmcv保持一致
        img_prefix = self.data_prefix.get('img_path', self.data_prefix.get('img', ''))
        T_img_path = osp.join(img_prefix, img_info['file_name']['Template'])
        S_img_path = osp.join(img_prefix, img_info['file_name']['Search'])
        if self.data_prefix.get('seg', None):                                # seg 前缀/割掩码
            T_seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name']['Template'].rsplit('.', 1)[0] + self.seg_map_suffix)
            S_seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name']['Search'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            T_seg_map_path = None
            S_seg_map_path = None
        data_info['T_img_path'] = T_img_path
        data_info['S_img_path'] = S_img_path
        data_info['img_id'] = img_info['img_id']
        data_info['T_seg_map_path'] = T_seg_map_path
        data_info['S_seg_map_path'] = S_seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = list(self.metainfo.get('classes', ()))
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            
            # Template
            x1, y1, w, h = ann['bbox']['Template']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area']['Template'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            T_bbox = [x1, y1, x1 + w, y1 + h]

            # Search
            x1, y1, w, h = ann['bbox']['Search']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area']['Search'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            S_bbox = [x1, y1, x1 + w, y1 + h]
            
            #密集标注
            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['T_bbox'] = T_bbox
            instance['S_bbox'] = S_bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            
            # 生成自适应文本提示/拆分为两句话
            use_adaptive = self.metainfo.get('adaptive_text_prompts', {}).get('enable', False)
            adaptive_text = None
            if use_adaptive:
                adaptive_text = self._generate_adaptive_text_prompt(S_bbox)
            base_text = 'uav .'
            instance['text'] = adaptive_text if adaptive_text else base_text

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        # 根据实例文本构建样本级文本提示，供模型直接读取
        instance_texts = []
        for instance in instances:
            text_value = instance.get('text')
            if text_value:
                str_value = str(text_value)
                if str_value not in instance_texts:
                    instance_texts.append(str_value)
        if not instance_texts:
            base_text = 'uav .'
            instance_texts = [base_text]
        data_info['text'] = instance_texts

        # ==== motion history for search frame: previous K GT boxes in same sequence ====
        try:
            search_rel = img_info['file_name']['Search']
            seq_rel_dir = osp.dirname(search_rel)
            cache = self._seq_motion_cache.get(seq_rel_dir, None)
            if cache is not None:
                frame_stem = osp.splitext(osp.basename(search_rel))[0]
                frame_idx = int(frame_stem) - 1  # filenames start at 1
                max_len = self.motion_history_max_len
                history = []
                history_frame_ids = []
                gt_rects = cache.get('gt_rect', [])
                exists = cache.get('exist', [])
                start = max(0, frame_idx - max_len)
                for idx in range(start, frame_idx):
                    if idx < len(gt_rects):
                        rect = gt_rects[idx]
                        if idx < len(exists) and exists[idx] and rect and len(rect) >= 4:
                            x1, y1, w, h = rect[:4]
                            if w > 0 and h > 0:
                                history.append([x1, y1, x1 + w, y1 + h])
                                history_frame_ids.append(idx)
                if history:
                    data_info['motion_history_gt'] = history
                    data_info['motion_history_gt_frame_ids'] = history_frame_ids
        except Exception:
            pass
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list) #去重
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
