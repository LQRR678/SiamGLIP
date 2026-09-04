# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmcv
import mmengine
import numpy as np
import torch
import torch.nn as nn
from datasets.pipelines import SiameseLoadImageFromFile
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import BaseInferencer, ModelType
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import _load_checkpoint_to_model
from mmengine.visualization import Visualizer
from rich.progress import track

from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import encode_mask_results, mask2bbox
from mmdet.utils import ConfigType
from mmdet.evaluation import get_classes

try:
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb
except ImportError:
    id2rgb = None
    VOID = None

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = List[DetDataSample]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')
DEBUG_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'debug_log.txt'))


class SiameseDetInferencer(BaseInferencer):
    """Object Detection Inferencer.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "rtmdet-s" or 'rtmdet_s_8xb32-300e_coco' or
            "configs/rtmdet/rtmdet_s_8xb32-300e_coco.py".
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to mmdet.
        palette (str): Color palette used for visualization. The order of
            priority is palette -> config -> checkpoint. Defaults to 'none'.
        show_progress (bool): Control whether to display the progress
            bar during the inference process. Defaults to True.
    """

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = {
        'return_vis',
        'show',
        'wait_time',
        'draw_pred',
        'pred_score_thr',
        'img_out_dir',
        'no_save_vis',
    }
    postprocess_kwargs: set = {
        'print_result',
        'pred_out_dir',
        'return_datasamples',
        'no_save_pred',
    }

    def __init__(self,
                 model: Optional[Union[ModelType, str]] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmdet',
                 palette: str = 'none',
                 show_progress: bool = True) -> None:
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0
        self.num_predicted_imgs = 0
        self.palette = palette
        init_default_scope(scope)
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)
        self.model = revert_sync_batchnorm(self.model)
        self.show_progress = show_progress
        # 保存device属性
        self._inferencer_device = device
    
    # def forward(self, inputs, **kwargs):
    #     """Forward pass for Siamese model.
        
    #     Args:
    #         inputs: Pipeline output dict or list of dicts containing 'T_inputs', 'S_inputs', 'data_samples'
            
    #     Returns:
    #         Predictions from the model
    #     """
    #     # 获取设备
    #     device = next(self.model.parameters()).device
        
    #     # 处理不同的输入格式
    #     if isinstance(inputs, dict):
    #         # 单个样本，pipeline直接返回的字典
    #         data_sample = inputs.get('data_samples', DetDataSample())
            
    #         # 如果data_sample是None或不是DetDataSample，创建一个新的
    #         if not isinstance(data_sample, DetDataSample):
    #             data_sample = DetDataSample()
            
    #         # 确保metainfo有必要的键
    #         metainfo = data_sample.metainfo if hasattr(data_sample, 'metainfo') else {}
            
    #         # 从inputs中提取并补充metainfo
    #         for key in ['T_img_shape', 'S_img_shape', 'T_scale_factor', 'S_scale_factor', 
    #                    'T_ori_shape', 'S_ori_shape', 'img_shape', 'ori_shape', 'scale_factor',
    #                    'T_img_path', 'S_img_path']:
    #             if key in inputs:
    #                 metainfo[key] = inputs[key]
            
    #         # 设置img_shape（bbox_head需要）- 优先使用T_img_shape
    #         if 'img_shape' not in metainfo:
    #             if 'T_img_shape' in metainfo:
    #                 metainfo['img_shape'] = metainfo['T_img_shape']
    #             else:
    #                 # 使用默认值
    #                 metainfo['img_shape'] = [800, 1333]
            
    #         if 'ori_shape' not in metainfo:
    #             if 'T_ori_shape' in metainfo:
    #                 metainfo['ori_shape'] = metainfo['T_ori_shape']
    #             else:
    #                 metainfo['ori_shape'] = [800, 1333]
            
    #         if 'scale_factor' not in metainfo:
    #             if 'T_scale_factor' in metainfo:
    #                 metainfo['scale_factor'] = metainfo['T_scale_factor']
    #             else:
    #                 metainfo['scale_factor'] = [1.0, 1.0]
            
    #         data_sample.set_metainfo(metainfo)
    #         batch_data_samples = [data_sample]
            
    #         # 提取T和S输入
    #         T_input = inputs['T_inputs']
    #         S_input = inputs['S_inputs']
            
    #         # 统一的tensor处理函数
    #         def to_batch_tensor(x):
    #             if isinstance(x, torch.Tensor):
    #                 # 确保是float类型
    #                 if x.dtype != torch.float32:
    #                     x = x.float()
    #                 # 确保有batch维度
    #                 if len(x.shape) == 3:  # (C, H, W)
    #                     return x.unsqueeze(0).to(device)
    #                 return x.to(device)
    #             elif isinstance(x, (list, tuple)) and len(x) > 0:
    #                 if isinstance(x[0], torch.Tensor):
    #                     stacked = torch.stack(x)
    #                     if stacked.dtype != torch.float32:
    #                         stacked = stacked.float()
    #                     return stacked.to(device)
    #                 else:
    #                     tensor = torch.stack([torch.tensor(t).float() for t in x])
    #                     return tensor.to(device)
    #             else:
    #                 tensor = torch.tensor(x).float()
    #                 if len(tensor.shape) == 3:
    #                     return tensor.unsqueeze(0).to(device)
    #                 return tensor.to(device)
            
    #         T_inputs = to_batch_tensor(T_input)
    #         S_inputs = to_batch_tensor(S_input)
    #     elif isinstance(inputs, (list, tuple)):
    #         # 多个样本
    #         batch_data_samples = []
    #         T_inputs_list = []
    #         S_inputs_list = []
            
    #         for data_dict in inputs:
    #             if not isinstance(data_dict, dict):
    #                 continue
    #             T_input = data_dict.get('T_inputs')
    #             S_input = data_dict.get('S_inputs')
    #             data_sample = data_dict.get('data_samples')
                
    #             if T_input is not None and S_input is not None:
    #                 T_inputs_list.append(T_input)
    #                 S_inputs_list.append(S_input)
    #                 if data_sample is not None:
    #                     batch_data_samples.append(data_sample)
    #                 else:
    #                     # 如果没有data_sample，创建一个空的
    #                     batch_data_samples.append(DetDataSample())
            
    #         if T_inputs_list:
    #             device = next(self.model.parameters()).device  # 获取模型设备
    #             T_inputs = torch.stack(T_inputs_list).to(device)
    #             S_inputs = torch.stack(S_inputs_list).to(device)
    #         else:
    #             raise ValueError(f"No valid inputs found. T_inputs_list length: {len(T_inputs_list)}")
    #     else:
    #         raise TypeError(f"Unsupported input type: {type(inputs)}")
        
        
    #     # 调用模型的forward
    #     with torch.no_grad():
    #         predictions = self.model(
    #             T_inputs=T_inputs,
    #             S_inputs=S_inputs,
    #             data_samples=batch_data_samples,
    #             mode='predict'
    #         )
        
    #     return predictions

    def _load_weights_to_model(self, model: nn.Module,
                               checkpoint: Optional[dict],
                               cfg: Optional[ConfigType]) -> None:
        """Loading model weights and meta information from cfg and checkpoint.

        Args:
            model (nn.Module): Model to load weights and meta information.
            checkpoint (dict, optional): The loaded checkpoint.
            cfg (Config or ConfigDict, optional): The loaded config.
        """

        if checkpoint is not None:
            _load_checkpoint_to_model(model, checkpoint)
            checkpoint_meta = checkpoint.get('meta', {})
            # save the dataset_meta in the model for convenience
            if 'dataset_meta' in checkpoint_meta:
                # mmdet 3.x, all keys should be lowercase
                model.dataset_meta = {
                    k.lower(): v
                    for k, v in checkpoint_meta['dataset_meta'].items()
                }
            elif 'CLASSES' in checkpoint_meta:
                # < mmdet 3.x
                classes = checkpoint_meta['CLASSES']
                model.dataset_meta = {'classes': classes}
            else:
                warnings.warn(
                    'dataset_meta or class names are not saved in the '
                    'checkpoint\'s meta data, use COCO classes by default.')
                model.dataset_meta = {'classes': get_classes('coco')}
        else:
            warnings.warn('Checkpoint is not loaded, and the inference '
                          'result is calculated by the randomly initialized '
                          'model!')
            warnings.warn('weights is None, use COCO classes by default.')
            model.dataset_meta = {'classes': get_classes('coco')}

        # Priority:  args.palette -> config -> checkpoint
        if self.palette != 'none':
            model.dataset_meta['palette'] = self.palette
        else:
            test_dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
            # lazy init. We only need the metainfo.
            test_dataset_cfg['lazy_init'] = True
            metainfo = DATASETS.build(test_dataset_cfg).metainfo
            cfg_palette = metainfo.get('palette', None)
            if cfg_palette is not None:
                model.dataset_meta['palette'] = cfg_palette
            else:
                if 'palette' not in model.dataset_meta:
                    warnings.warn(
                        'palette does not exist, random is used by default. '
                        'You can also set the palette to customize.')
                    model.dataset_meta['palette'] = 'random'

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if 'meta_keys' in pipeline_cfg[-1]:
            pipeline_cfg[-1]['meta_keys'] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
                if meta_key != 'img_id')
        # import pdb;pdb.set_trace()
        load_img_idx = self._get_transform_idx(
            pipeline_cfg, ('SiameseLoadImageFromFile', SiameseLoadImageFromFile))
        # import pdb;pdb.set_trace()
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        # pipeline_cfg[load_img_idx]['type'] = 'mmdet.InferencerLoader'
        # import pdb;pdb.set_trace()
        return Compose(pipeline_cfg)

    def _get_transform_idx(self, pipeline_cfg: ConfigType,
                           name: Union[str, Tuple[str, type]]) -> int:
        """Returns the index of the transform in a pipeline.

        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] in name:
                return i
        return -1

    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        """Initialize visualizers.

        Args:
            cfg (ConfigType): Config containing the visualizer information.

        Returns:
            Visualizer or None: Visualizer initialized with config.
        """
        visualizer = super()._init_visualizer(cfg)
        visualizer.dataset_meta = self.model.dataset_meta
        return visualizer

    def _inputs_to_list(self, inputs: InputsType) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - str:
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                filename_list = list_dir_or_file(
                    inputs, list_dir=False, suffix=IMG_EXTENSIONS)
                inputs = [
                    join_path(inputs, filename) for filename in filename_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _build_siamese_input_dict(self, template_input: InputType,
                                  search_input: InputType) -> dict:
        """Ensure each sample is represented as a dict with T/S entries."""
        if isinstance(template_input, dict):
            data = copy.deepcopy(template_input)
        else:
            data = {}
            if isinstance(template_input, str):
                data['T_img_path'] = template_input
            else:
                data['T_img'] = template_input

        if isinstance(search_input, dict):
            if 'S_img_path' in search_input and \
                    'S_img_path' not in data:
                data['S_img_path'] = search_input['S_img_path']
            if 'S_img' in search_input and 'S_img' not in data:
                data['S_img'] = search_input['S_img']
        else:
            if isinstance(search_input, str):
                data.setdefault('S_img_path', search_input)
            else:
                data.setdefault('S_img', search_input)

        if 'T_img_path' not in data and 'T_img' not in data:
            raise ValueError('Template branch input is missing.')
        if 'S_img_path' not in data and 'S_img' not in data:
            raise ValueError('Search branch input is missing.')
        return data

    def _flatten_template_bbox_input(
            self, template_bboxes: Union[Sequence, np.ndarray, torch.Tensor,
                                         Dict]) -> List:
        """Convert raw template_bboxes input into a list."""
        if isinstance(template_bboxes, torch.Tensor):
            template_bboxes = template_bboxes.detach().cpu().tolist()
        elif isinstance(template_bboxes, np.ndarray):
            template_bboxes = template_bboxes.tolist()

        if template_bboxes is None:
            return []

        if isinstance(template_bboxes, dict):
            return [template_bboxes]

        if isinstance(template_bboxes, (list, tuple)):
            if len(template_bboxes) == 0:
                return []
            first_elem = template_bboxes[0]
            if isinstance(first_elem, (list, tuple, dict, np.ndarray,
                                       torch.Tensor)):
                return list(template_bboxes)
            return [template_bboxes]

        return [template_bboxes]

    def _coerce_bbox_to_list(self, bbox: Union[Sequence, np.ndarray,
                                               torch.Tensor]) -> List[float]:
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().flatten().tolist()
        elif isinstance(bbox, np.ndarray):
            bbox = bbox.flatten().tolist()

        if not isinstance(bbox, (list, tuple)):
            raise ValueError('Each template bbox must be a sequence of 4 values.')
        if len(bbox) != 4:
            raise ValueError('Each template bbox must have exactly 4 values.')
        return [float(v) for v in bbox]

    def _prepare_template_instances(
            self,
            template_bboxes: Optional[Union[Sequence, np.ndarray, torch.Tensor,
                                            Dict]],
            num_samples: int) -> Optional[List[Optional[dict]]]:
        if template_bboxes is None:
            return None

        flattened = self._flatten_template_bbox_input(template_bboxes)
        if not flattened:
            return None

        if len(flattened) not in (1, num_samples):
            raise ValueError('Number of template_bboxes must match the number '
                             'of inputs or be a single bbox for broadcasting.')
        if len(flattened) == 1 and num_samples > 1:
            flattened = [copy.deepcopy(flattened[0]) for _ in range(num_samples)]

        processed: List[Optional[dict]] = []
        for entry in flattened:
            if entry is None:
                processed.append(None)
                continue
            if isinstance(entry, dict):
                bbox_values = entry.get('bbox') or entry.get('T_bbox') or entry.get(
                    'S_bbox')
                if bbox_values is None:
                    raise ValueError('Template bbox dict must contain "bbox", '
                                     '"T_bbox" or "S_bbox".')
                label = entry.get('bbox_label', 0)
                ignore_flag = entry.get('ignore_flag', 0)
            else:
                bbox_values = entry
                label = 0
                ignore_flag = 0
            bbox_list = self._coerce_bbox_to_list(bbox_values)
            processed.append({
                'bbox': bbox_list,
                'bbox_label': int(label),
                'ignore_flag': int(ignore_flag)
            })
        return processed

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, batch_size)
        yield from map(self.collate_fn, chunked_data)

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    if isinstance(inputs_, dict):
                        # 正确处理T和S输入
                        if 'T_img' in inputs_:
                            T_ori_inputs_ = inputs_['T_img']
                            S_ori_inputs_ = inputs_['S_img']
                        else:
                            T_ori_inputs_ = inputs_['T_img_path']
                            S_ori_inputs_ = inputs_['S_img_path']
                        if isinstance(inputs_.get('T_img_path'), str) and \
                                isinstance(inputs_.get('S_img_path'), str):
                            same_flag = False
                            try:
                                same_flag = os.path.samefile(
                                    inputs_['T_img_path'],
                                    inputs_['S_img_path'])
                            except Exception:
                                same_flag = (inputs_['T_img_path'] ==
                                             inputs_['S_img_path'])
                            # try:
                            #     with open(DEBUG_LOG_PATH, 'a') as dbg:
                            #         print('[infer-debug] chunk_data '
                            #               f"T_path={inputs_['T_img_path']} "
                            #               f"S_path={inputs_['S_img_path']} "
                            #               f'same_file={same_flag}',
                            #               file=dbg)
                            # except OSError:
                            #     pass
                        chunk_data.append(
                            ((T_ori_inputs_, S_ori_inputs_),
                             self.pipeline(copy.deepcopy(inputs_))))
                    else:
                        chunk_data.append((inputs_, self.pipeline(inputs_)))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    def __call__(
            self,
            T_inputs: InputsType,
            S_inputs: InputsType,
            batch_size: int = 1,
            fusion_bypass: bool = False,
            return_vis: bool = False,
            show: bool = False,
            wait_time: int = 0,
            no_save_vis: bool = False,
            draw_pred: bool = True,
            pred_score_thr: float = 0.3,
            return_datasamples: bool = False,
            print_result: bool = False,
            no_save_pred: bool = True,
            out_dir: str = '',
            motion_history: Optional[Union[Sequence, np.ndarray, torch.Tensor, Dict]] = None,
            motion_history_frame_ids: Optional[Union[Sequence, np.ndarray, torch.Tensor, Dict]] = None,
            motion_history_format: str = 'xyxy',
            # by open image task
            texts: Optional[Union[str, list]] = None,
            # by open panoptic task
            stuff_texts: Optional[Union[str, list]] = None,
            # by GLIP and Grounding DINO
            custom_entities: bool = False,
            template_bboxes: Optional[Union[Sequence, np.ndarray, torch.Tensor,
                                            Dict]] = None,
            # by Grounding DINO
            tokens_positive: Optional[Union[int, list]] = None,
            **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            return_datasamples (bool): Whether to return results as
                :obj:`DetDataSample`. Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to True.
            out_dir: Dir to save the inference results or
                visualization. If left as empty, no file will be saved.
                Defaults to ''.
            texts (str | list[str]): Text prompts. Defaults to None.
            stuff_texts (str | list[str]): Stuff text prompts of open
                panoptic task. Defaults to None.
            custom_entities (bool): Whether to use custom entities.
                Defaults to False. Only used in GLIP and Grounding DINO.
            template_bboxes (Sequence | np.ndarray | torch.Tensor | dict |
                None): Template bbox information in ``[x1, y1, x2, y2]`` format
                used to build ``instances`` during inference so that template
                branch can access ``gt_instances`` like in training.
            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        # 只处理Template分支，分支融合请额外设计
        T_ori_inputs = self._inputs_to_list(T_inputs)
        S_ori_inputs = self._inputs_to_list(S_inputs)
        if len(T_ori_inputs) != len(S_ori_inputs):
            raise ValueError('Template and search inputs must have the same '
                             'number of samples.')

        if texts is not None and isinstance(texts, str):
            texts = [texts] * len(T_ori_inputs)
        if stuff_texts is not None and isinstance(stuff_texts, str):
            stuff_texts = [stuff_texts] * len(T_ori_inputs)
        if motion_history is not None and not isinstance(motion_history, (list, tuple)):
            motion_history = [motion_history] * len(T_ori_inputs)
        if motion_history_frame_ids is not None and not isinstance(
                motion_history_frame_ids, (list, tuple)):
            motion_history_frame_ids = [motion_history_frame_ids
                                        ] * len(T_ori_inputs)

        # Currently only supports bs=1
        tokens_positive = [tokens_positive] * len(T_ori_inputs)

        siamese_inputs: List[dict] = []
        for temp_item, search_item in zip(T_ori_inputs, S_ori_inputs):
            siamese_inputs.append(
                self._build_siamese_input_dict(temp_item, search_item))
        T_ori_inputs = siamese_inputs

        if texts is not None:
            assert len(texts) == len(T_ori_inputs)
            for i in range(len(texts)):
                prompt = texts[i]
                T_ori_inputs[i]['text'] = prompt
                T_ori_inputs[i]['test'] = prompt
                T_ori_inputs[i]['custom_entities'] = custom_entities
                T_ori_inputs[i]['tokens_positive'] = tokens_positive[i]
        if stuff_texts is not None:
            assert len(stuff_texts) == len(T_ori_inputs)
            for i in range(len(stuff_texts)):
                T_ori_inputs[i]['stuff_text'] = stuff_texts[i]

        template_instances = self._prepare_template_instances(
            template_bboxes, len(T_ori_inputs))
        if template_instances is not None:
            for idx, instance in enumerate(template_instances):
                if instance is None:
                    continue
                instance_list = T_ori_inputs[idx].setdefault('instances', [])
                if not isinstance(instance_list, list):
                    raise TypeError('`instances` must be a list when provided.')
                instance_list.append(copy.deepcopy(instance))

        # attach motion history to samples
        if motion_history is not None:
            if len(motion_history) not in (1, len(T_ori_inputs)):
                raise ValueError('Number of motion_history entries must match inputs or be 1.')
            if len(motion_history) == 1 and len(T_ori_inputs) > 1:
                motion_history = [copy.deepcopy(motion_history[0]) for _ in range(len(T_ori_inputs))]
        if motion_history_frame_ids is not None:
            if len(motion_history_frame_ids) not in (1, len(T_ori_inputs)):
                raise ValueError('Number of motion_history_frame_ids entries must match inputs or be 1.')
            if len(motion_history_frame_ids) == 1 and len(T_ori_inputs) > 1:
                motion_history_frame_ids = [
                    copy.deepcopy(motion_history_frame_ids[0])
                    for _ in range(len(T_ori_inputs))
                ]
        if motion_history is not None:
            for i in range(len(T_ori_inputs)):
                T_ori_inputs[i]['motion_history'] = motion_history[i]
                T_ori_inputs[i]['motion_history_format'] = motion_history_format
                if motion_history_frame_ids is not None:
                    T_ori_inputs[i]['motion_history_frame_ids'] = motion_history_frame_ids[i]

        T_inputs = self.preprocess(T_ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {'predictions': [], 'visualization': []}
        for (T_ori_imgs, S_ori_imgs), data in (track(T_inputs, description='Inference')
                               if self.show_progress else T_inputs):
            # 将fusion_bypass注入到模型（一次性设置）
            try:
                if hasattr(self.model, 'fusion_bypass'):
                    self.model.fusion_bypass = fusion_bypass
            except Exception:
                pass
            preds = self.forward(data, **forward_kwargs)
            visualization = self.visualize(
                S_ori_imgs,  # 使用S图像进行可视化
                preds,
                return_vis=return_vis,
                show=show,
                wait_time=wait_time,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                no_save_vis=no_save_vis,
                img_out_dir=out_dir,
                **visualize_kwargs)
            results = self.postprocess(
                preds,
                visualization,
                return_datasamples=return_datasamples,
                print_result=print_result,
                no_save_pred=no_save_pred,
                pred_out_dir=out_dir,
                **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions'])
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])
        return results_dict

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  **kwargs) -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, 'vis',
                                img_name) if img_out_dir != '' else None

            self.visualizer.add_datasample(
                img_name,
                img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results

    def postprocess(
        self,
        preds: PredType,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasamples: bool = False,
        print_result: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = '',
        **kwargs,
    ) -> Dict:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            preds (List[:obj:`DetDataSample`]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            return_datasamples (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to False.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
                :meth:`forward` and processed in :meth:`postprocess`.
                If ``return_datasamples=False``, it usually should be a
                json-serializable dict containing only basic data elements such
                as strings and numbers.
        """
        if no_save_pred is True:
            pred_out_dir = ''

        result_dict = {}
        results = preds
        if not return_datasamples:
            results = []
            for pred in preds:
                result = self.pred2dict(pred, pred_out_dir)
                results.append(result)
        elif pred_out_dir != '':
            warnings.warn('Currently does not support saving datasample '
                          'when return_datasamples is set to True. '
                          'Prediction results are not saved!')
        # Add img to the results after printing and dumping
        result_dict['predictions'] = results
        if print_result:
            print(result_dict)
        result_dict['visualization'] = visualization
        return result_dict

    # TODO: The data format and fields saved in json need further discussion.
    #  Maybe should include model name, timestamp, filename, image info etc.
    def pred2dict(self,
                  data_sample: DetDataSample,
                  pred_out_dir: str = '') -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        is_save_pred = True
        if pred_out_dir == '':
            is_save_pred = False

        if is_save_pred and 'img_path' in data_sample:
            img_path = osp.basename(data_sample.img_path)
            img_path = osp.splitext(img_path)[0]
            out_img_path = osp.join(pred_out_dir, 'preds',
                                    img_path + '_panoptic_seg.png')
            out_json_path = osp.join(pred_out_dir, 'preds', img_path + '.json')
        elif is_save_pred:
            out_img_path = osp.join(
                pred_out_dir, 'preds',
                f'{self.num_predicted_imgs}_panoptic_seg.png')
            out_json_path = osp.join(pred_out_dir, 'preds',
                                     f'{self.num_predicted_imgs}.json')
            self.num_predicted_imgs += 1

        result = {}
        if 'pred_instances' in data_sample:
            masks = data_sample.pred_instances.get('masks')
            pred_instances = data_sample.pred_instances.numpy()
            result = {
                'labels': pred_instances.labels.tolist(),
                'scores': pred_instances.scores.tolist()
            }
            if 'bboxes' in pred_instances:
                result['bboxes'] = pred_instances.bboxes.tolist()
            if masks is not None:
                if 'bboxes' not in pred_instances or pred_instances.bboxes.sum(
                ) == 0:
                    # Fake bbox, such as the SOLO.
                    bboxes = mask2bbox(masks.cpu()).numpy().tolist()
                    result['bboxes'] = bboxes
                encode_masks = encode_mask_results(pred_instances.masks)
                for encode_mask in encode_masks:
                    if isinstance(encode_mask['counts'], bytes):
                        encode_mask['counts'] = encode_mask['counts'].decode()
                result['masks'] = encode_masks

        if 'pred_panoptic_seg' in data_sample:
            if VOID is None:
                raise RuntimeError(
                    'panopticapi is not installed, please install it by: '
                    'pip install git+https://github.com/cocodataset/'
                    'panopticapi.git.')

            pan = data_sample.pred_panoptic_seg.sem_seg.cpu().numpy()[0]
            pan[pan % INSTANCE_OFFSET == len(
                self.model.dataset_meta['classes'])] = VOID
            pan = id2rgb(pan).astype(np.uint8)

            if is_save_pred:
                mmcv.imwrite(pan[:, :, ::-1], out_img_path)
                result['panoptic_seg_path'] = out_img_path
            else:
                result['panoptic_seg'] = pan

        if is_save_pred:
            mmengine.dump(result, out_json_path)

        return result
