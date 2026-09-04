import glob
import json
import os
import os.path as osp

import mmcv
import numpy as np
from tqdm import tqdm


FRAME_INTERVAL = 20
# mode = "train"
mode = "val"

adenomatous_json_dir = f"/to/Anti-UAV410/{mode}"
# adenomatous_json_dir = f"/to/Anti-UAVALLFinal"
out_path = "annotationsantiuav_Fianl/"
os.makedirs(out_path, exist_ok=True)
out_json = osp.join(out_path, f"{mode}.json")

merged_data = {
    "licenses": [{"name": "", "id": 0, "url": ""}],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": "",
    },
    "categories": [{"id": 1, "name": "uav", "supercategory": ""}],
    "images": [],
    "annotations": [],
}


def is_valid_bbox(bbox):
    """Check bbox is [x, y, w, h] with positive width/height."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    _, _, w, h = bbox
    return w > 0 and h > 0


def collect_valid_indices(exist, annos):
    return [
        idx
        for idx, (flag, bbox) in enumerate(zip(exist, annos))
        if flag == 1 and is_valid_bbox(bbox)
    ]


sequences = [
    d for d in sorted(os.listdir(adenomatous_json_dir)) if osp.isdir(osp.join(adenomatous_json_dir, d))
]

for seq in tqdm(sequences):
    json_path = osp.join(adenomatous_json_dir, seq, "IR_label.json")
    if not osp.isfile(json_path):
        continue

    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    exist = data["exist"]
    annos = data["gt_rect"]
    image_files = sorted(glob.glob(osp.join(adenomatous_json_dir, seq, "*.jpg")))

    if len(image_files) != len(annos):
        continue

    valid_indices = collect_valid_indices(exist, annos)
    if len(valid_indices) < 2:
        continue

    for template_idx in valid_indices[::FRAME_INTERVAL]:
        search_candidates = [idx for idx in valid_indices if idx != template_idx]
        if not search_candidates:
            continue

        search_idx = int(np.random.choice(search_candidates, 1)[0])
        template_rel = osp.join(seq, f"{template_idx + 1:06d}.jpg")
        search_rel = osp.join(seq, f"{search_idx + 1:06d}.jpg")
        template_abs = osp.join(adenomatous_json_dir, template_rel)
        search_abs = osp.join(adenomatous_json_dir, search_rel)

        if not (osp.isfile(template_abs) and osp.isfile(search_abs)):
            continue

        img_ori = mmcv.imread(template_abs)
        if img_ori is None:
            continue

        image_entry = {
            "id": len(merged_data["images"]) + 1,
            "file_name": {"Template": template_rel, "Search": search_rel},
            "height": img_ori.shape[0],
            "width": img_ori.shape[1],
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }
        merged_data["images"].append(image_entry)

        Tbbox = annos[template_idx]
        Sbbox = annos[search_idx]

        annotation_entry = {
            "id": len(merged_data["annotations"]) + 1,
            "category_id": 1,
            "image_id": image_entry["id"],
            "bbox": {"Template": Tbbox, "Search": Sbbox},
            "segmentation": [],
            "area": {"Template": Tbbox[2] * Tbbox[3], "Search": Sbbox[2] * Sbbox[3]},
            "iscrowd": 0,
            "attributes": {"occluded": False},
        }
        merged_data["annotations"].append(annotation_entry)

print(f"images {len(merged_data['images'])}, annos {len(merged_data['annotations'])}")

with open(out_json, "w") as out_file:
    json.dump(merged_data, out_file)
