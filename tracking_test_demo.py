import json
import glob
import os
import contextlib
import io
import torch
import tqdm
import numpy as np
from PIL import Image

from trackers import SiameseDetInferencer
import init_paths
from mmengine.config import Config
from typing import Dict, List, Optional, Tuple, Union
from mmdet.structures import DetDataSample, OptSampleList, SampleList

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# source activate siamglip
# CUDA_VISIBLE_DEVICES=0 python tracking_test_demo.py 

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def not_exist(pred):

    if len(pred) == 1 or len(pred) == 0:
        return 1.0
    else:
        return 0.0

def eval(out_res,  label_res):

    measure_per_frame = []

    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):

        if not _exist:
            measure_per_frame.append(not_exist(_pred))
        else:

            if len(_gt) < 4 or sum(_gt) == 0:
                continue

            if len(_pred) == 4:
                measure_per_frame.append(iou(_pred, _gt))
            else:
                measure_per_frame.append(0.0)

            # try:
            #     measure_per_frame.append(iou(_pred, _gt))
            # except:
            #     measure_per_frame.append(0)

        # measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt))

    return np.mean(measure_per_frame)


config_file = 'configs/tracking/siamlip_swin_tiny_fpn.py'

_cfg = Config.fromfile(config_file)
motion_history_max_len = _cfg.get('motion_history_max_len')

checkpoint_file = 'work_dirs/siamglip_swin_tiny_fpn.pth'
device = 'cuda:0'

dataset_dir = '/to/Anti-UAV410/Anti-UAV/test/'
# dataset_dir = '/to/Anti-UAVALLFinal/test/'
output_dir = './reslts' 
prompt_output_dir = os.path.join(output_dir, 'prompts')


# def get_last_prompt_entries(model):
#     tracking_model = model
#     while hasattr(tracking_model, 'module'):
#         tracking_model = tracking_model.module

#     getter = getattr(tracking_model, 'get_last_prompt_structures', None)
#     if getter is None:
#         return []

#     batch_structures = getter()
#     if not batch_structures:
#         return []
#     return batch_structures[0]

inferencer = SiameseDetInferencer(
    model=config_file,
    weights=checkpoint_file,
    device=device,
    palette='coco'  # 可选：voc, citys, random, none
)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(prompt_output_dir, exist_ok=True)


anno_files = sorted(glob.glob(
            os.path.join(dataset_dir, '*/IR_label.json')))

seq_dirs = [os.path.dirname(f) for f in anno_files]
seq_dirs = seq_dirs[0:]

overall_performance = []
s=0

for seq_dir in tqdm.tqdm(seq_dirs):

    seq_name = os.path.basename(seq_dir)
    inferencer.model.reset_tracking_state()

    # skip if results exist
    record_file = os.path.join(
        output_dir, '%s.txt' % seq_name)
    prompt_record_file = os.path.join(
        prompt_output_dir, '%s.json' % seq_name)
    if os.path.exists(record_file) and os.path.exists(prompt_record_file):
        print('  Found results and prompt log, skipping', seq_name)
        continue

    img_files = sorted(glob.glob(
        os.path.join(seq_dir, '*.*g')))

    json_dir = os.path.join(seq_dir, 'IR_label.json')

    with open(json_dir) as json_file:
        data = json.load(json_file)
        annos = data['gt_rect']

    seq_rel = os.path.relpath(seq_dir, os.path.dirname(os.path.normpath(dataset_dir)))
    cache_key = seq_rel.replace('\\', '/')

    # sequence_text_prompt
    first_bbox = np.array(annos[0], dtype=np.float64)
    template_bbox_xyxy = None
    if first_bbox.size >= 4:
        template_bbox_xyxy = [
            float(first_bbox[0]),
            float(first_bbox[1]),
            float(first_bbox[0] + first_bbox[2]),
            float(first_bbox[1] + first_bbox[3])
        ]
   

    frame_num = len(img_files)
    bboxes = np.zeros((frame_num, 4))
    history_xyxy = []
    history_frame_ids = []
    prompt_frames = []
    pred_bbox_format = 'auto'  # inferencer 预测框的格式（auto/xywh/xyxy）

    debug_log_path = os.path.join(os.path.dirname(__file__), 'debug_log.txt')

    for f, img_file in enumerate(img_files):
        try:
            frame_id = int(os.path.splitext(os.path.basename(img_file))[0]) - 1
        except Exception:
            frame_id = f

        # first frame
        if f == 0:
            bbox = annos[0]
            template_img_file = img_file
            gt_xyxy = [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[0] + bbox[2] - 1), 
                float(bbox[1] + bbox[3] - 1)
            ]
            history_xyxy.append(gt_xyxy)
            history_frame_ids.append(frame_id)
            prompt_frames.append({
                'frame_index': f,
                'frame_id': frame_id,
                'image_file': os.path.basename(img_file),
                'is_initialization': True,
                'motion_history_frame_ids': [],
                'prompts': [],
                'note': 'Initialization frame; no model inference prompt was used.',
            })
        else:
            motion_history_list = history_xyxy[-motion_history_max_len:]
            motion_history_frame_ids = history_frame_ids[-motion_history_max_len:]
            # ensure frame_ids length matches history
            if len(motion_history_frame_ids) == 0:
                motion_history_frame_ids = list(range(len(motion_history_list)))
            elif len(motion_history_frame_ids) != len(motion_history_list):
                # fallback: align lengths by padding/truncating with last known id
                if len(motion_history_frame_ids) > len(motion_history_list):
                    motion_history_frame_ids = motion_history_frame_ids[-len(motion_history_list):]
                else:
                    last_id = motion_history_frame_ids[-1] if motion_history_frame_ids else 0
                    while len(motion_history_frame_ids) < len(motion_history_list):
                        last_id += 1
                        motion_history_frame_ids.append(last_id)

            mf = io.StringIO()
            with contextlib.redirect_stdout(mf):
                result = inferencer(
                    T_inputs=template_img_file,
                    S_inputs=img_file,
                    template_bboxes=template_bbox_xyxy,
                    motion_history=[motion_history_list],
                    motion_history_frame_ids=[motion_history_frame_ids],
                    motion_history_format='xyxy',
                    pred_score_thr=0.0,
                    show=False,           # 是否弹出窗口显示
                    no_save_vis=True,   # 是否保存可视化结果
                    no_save_pred=True,  # 是否保存预测json
                    out_dir='outputs',    # 保存目录
                )

            # prompt_frames.append({
            #     'frame_index': f,
            #     'frame_id': frame_id,
            #     'image_file': os.path.basename(img_file),
            #     'is_initialization': False,
            #     'motion_history_frame_ids': [
            #         int(v) for v in motion_history_frame_ids
            #     ],
            #     'prompts': get_last_prompt_entries(inferencer.model),
            # })

            pred_list = result['predictions'][0].get('bboxes', [])
            if pred_list is None or len(pred_list) == 0:
                # no prediction: reuse last or fallback to zeros
                if history_xyxy:
                    pred_xyxy = history_xyxy[-1]
                    pred_xywh = [
                        pred_xyxy[0],
                        pred_xyxy[1],
                        pred_xyxy[2] - pred_xyxy[0] + 1,
                        pred_xyxy[3] - pred_xyxy[1] + 1,
                    ]
                else:
                    pred_xyxy = [0.0, 0.0, 0.0, 0.0]
                    pred_xywh = [0.0, 0.0, 0.0, 0.0]
            else:
                mmbbox = [float(v) for v in pred_list[0]]
                fmt = pred_bbox_format
                if fmt == 'auto':
                    fmt = 'xyxy' if mmbbox[2] > mmbbox[0] and mmbbox[3] > mmbbox[1] else 'xywh'
                if fmt == 'xywh':
                    pred_xywh = mmbbox
                    pred_xyxy = [
                        pred_xywh[0],
                        pred_xywh[1],
                        pred_xywh[0] + pred_xywh[2]-1,  # align with training closed-interval
                        pred_xywh[1] + pred_xywh[3]-1,
                    ]
                elif fmt == 'xyxy':
                    pred_xyxy = [
                        mmbbox[0],
                        mmbbox[1],
                        mmbbox[2]-1, 
                        mmbbox[3]-1  
                    ]
                    pred_xywh = [
                        mmbbox[0],
                        mmbbox[1],
                        mmbbox[2]-mmbbox[0],
                        mmbbox[3]-mmbbox[1],
                    ]

            history_xyxy.append(pred_xyxy)
            history_frame_ids.append(frame_id)
            
            if len(history_xyxy) > motion_history_max_len:
                history_xyxy = history_xyxy[-motion_history_max_len:]
                history_frame_ids = history_frame_ids[-motion_history_max_len:]

            bbox = np.array(pred_xywh)
            # import pdb;pdb.set_trace()
        bboxes[f, :] = bbox
    save_bboxes=bboxes.tolist()

    mixed_measure = eval(save_bboxes, data)
    overall_performance.append(mixed_measure)
    text = '[%03d/%03d] %20s %5s Fixed Measure: %.04f' % (s + 1, len(seq_dirs), seq_name, 'IR', mixed_measure)
    print(text)

    record_file = os.path.join(
            output_dir, '%s.txt' % seq_name)

    with open(record_file, 'w') as f:
        json.dump({'res': save_bboxes}, f)

    with open(prompt_record_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'sequence': seq_name,
                'config_file': config_file,
                'checkpoint_file': checkpoint_file,
                'motion_history_max_len': motion_history_max_len,
                'frames': prompt_frames,
            },
            f,
            ensure_ascii=False,
            indent=2)
    # print('  Results recorded at', record_file)

    s=s+1
text='[Overall] %5s Mixed Measure: %.04f\n' % ('IR', np.mean(overall_performance))
print(text)

print('finished!')
