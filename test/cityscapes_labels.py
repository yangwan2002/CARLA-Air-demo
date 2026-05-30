"""Cityscapes 19-class palette (trainId 0–18) for visualization."""

from __future__ import annotations

import numpy as np

# id2label from nvidia/segformer-b2-finetuned-cityscapes-512-1024
ID2LABEL = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

# Official Cityscapes color map (RGB), indexed by trainId
PALETTE = np.array(
    [
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ],
    dtype=np.uint8,
)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert HxW label map (trainIds) to HxWx3 RGB uint8."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for tid in range(len(PALETTE)):
        out[mask == tid] = PALETTE[tid]
    return out
