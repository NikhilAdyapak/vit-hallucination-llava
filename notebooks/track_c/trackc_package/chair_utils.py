"""
CHAIR object extraction — aligned with Track A notebook (TRACKA_1ST_INFERENCE_CHAIR.ipynb).
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

COCO_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

SYNONYMS = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "child": "person",
    "kid": "person",
    "baby": "person",
    "guy": "person",
    "people": "person",
    "player": "person",
    "rider": "person",
    "pedestrian": "person",
    "skier": "person",
    "snowboarder": "person",
    "surfer": "person",
    "lady": "person",
    "gentleman": "person",
    "individual": "person",
    "bike": "bicycle",
    "cycle": "bicycle",
    "automobile": "car",
    "vehicle": "car",
    "sedan": "car",
    "suv": "car",
    "motorbike": "motorcycle",
    "plane": "airplane",
    "aircraft": "airplane",
    "jet": "airplane",
    "ship": "boat",
    "vessel": "boat",
    "sailboat": "boat",
    "yacht": "boat",
    "stoplight": "traffic light",
    "signal light": "traffic light",
    "hydrant": "fire hydrant",
    "sofa": "couch",
    "loveseat": "couch",
    "television": "tv",
    "monitor": "tv",
    "screen": "tv",
    "notebook computer": "laptop",
    "computer": "laptop",
    "phone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "fridge": "refrigerator",
    "icebox": "refrigerator",
    "table": "dining table",
    "desk": "dining table",
    "counter": "dining table",
    "countertop": "dining table",
    "plant": "potted plant",
    "houseplant": "potted plant",
    "flower": "potted plant",
    "flower pot": "potted plant",
    "teddy": "teddy bear",
    "doughnut": "donut",
    "hotdog": "hot dog",
    "racquet": "tennis racket",
    "racket": "tennis racket",
    "purse": "handbag",
    "clutch": "handbag",
    "bag": "backpack",
    "rucksack": "backpack",
    "mug": "cup",
    "coffee cup": "cup",
    "glass": "wine glass",
    "goblet": "wine glass",
    "stove": "oven",
    "dryer": "hair drier",
    "hair dryer": "hair drier",
    "ball": "sports ball",
    "baseball": "sports ball",
    "basketball": "sports ball",
    "football": "sports ball",
    "soccer ball": "sports ball",
    "tennis ball": "sports ball",
    "bat": "baseball bat",
    "glove": "baseball glove",
    "mitt": "baseball glove",
    "ski": "skis",
    "disc": "frisbee",
    "parasol": "umbrella",
    "necktie": "tie",
    "bowtie": "tie",
    "luggage": "suitcase",
    "briefcase": "suitcase",
}


def extract_mentioned_objects(caption: str) -> set[str]:
    caption_lower = caption.lower()
    mentioned: set[str] = set()
    for obj in COCO_OBJECTS:
        if re.search(r"\b" + re.escape(obj) + r"\b", caption_lower):
            mentioned.add(obj)
    for synonym, coco_cat in SYNONYMS.items():
        if re.search(r"\b" + re.escape(synonym) + r"\b", caption_lower):
            mentioned.add(coco_cat)
    return mentioned


def compute_chair(caption: str, gt_objects: set[str]) -> dict[str, Any]:
    mentioned = extract_mentioned_objects(caption)
    if len(mentioned) == 0:
        return {
            "chair_i": 0.0,
            "is_hallucinating": False,
            "mentioned": set(),
            "hallucinated": set(),
            "n_mentioned": 0,
            "n_hallucinated": 0,
        }
    hallucinated = mentioned - gt_objects
    chair_i = len(hallucinated) / len(mentioned)
    return {
        "chair_i": chair_i,
        "is_hallucinating": len(hallucinated) > 0,
        "mentioned": mentioned,
        "hallucinated": hallucinated,
        "n_mentioned": len(mentioned),
        "n_hallucinated": len(hallucinated),
    }


def load_gt_lookup_from_coco(ann_path: str) -> dict[int, set[str]]:
    from pycocotools.coco import COCO

    coco = COCO(ann_path)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}

    def get_gt(image_id: int) -> set[str]:
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        return {cat_id_to_name[a["category_id"]] for a in anns}

    ids = coco.getImgIds()
    return {i: get_gt(i) for i in ids}


def chair_scores_for_captions(
    df: pd.DataFrame,
    gt_by_image_id: dict[int, set[str]],
    caption_col: str = "caption",
    image_id_col: str = "image_id",
) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        iid = int(r[image_id_col])
        cap = str(r[caption_col])
        gt = gt_by_image_id.get(iid, set())
        ch = compute_chair(cap, gt)
        rows.append(
            {
                image_id_col: iid,
                "chair_i": round(ch["chair_i"], 4),
                "is_hallucinating": ch["is_hallucinating"],
                "n_mentioned": ch["n_mentioned"],
                "n_hallucinated": ch["n_hallucinated"],
            }
        )
    return pd.DataFrame(rows)
