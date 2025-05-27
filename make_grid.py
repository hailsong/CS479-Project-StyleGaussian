#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather key frames and build grids.
Assumes this script is placed at—or above—the project root that contains:
./output/saengsang/artisctic/default/camera_style_*/

Requires: Python ≥ 3.8, OpenCV-Python (pip install opencv-python).
"""

import math
import shutil
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------
# 0) Locate the directory containing all camera_style_* folders
# ----------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent                # where this script lives
DATA_ROOT      = PROJECT_ROOT / "output" / "saengsaeng" / "artistic" / "default"
EXAMPLES_DIR   = DATA_ROOT / "examples"                         # target for copies & grids
TARGET_FRAMES  = ["0000.png", "0050.png", "0100.png"]

EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1) Copy the requested frames
# ----------------------------------------------------------------------
camera_dirs = sorted(DATA_ROOT.glob("camera_style_*"))          # all style folders
print(DATA_ROOT)
for cam in camera_dirs:
    for fname in TARGET_FRAMES:
        src = cam / fname
        if src.is_file():
            dst_name = f"{cam.name}_{fname}"                    # avoid overwriting
            shutil.copy2(src, EXAMPLES_DIR / dst_name)

# ----------------------------------------------------------------------
# 2) Build a grid for each frame index
# ----------------------------------------------------------------------
def build_grid(images):
    """Return a white-padded grid arranged as close to square as possible."""
    n    = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = images[0].shape[:2]
    grid = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)  # white background

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return grid

for fname in TARGET_FRAMES:
    imgs = []
    for cam in camera_dirs:
        p = cam / fname
        if p.is_file():
            imgs.append(cv2.imread(str(p), cv2.IMREAD_COLOR))

    if imgs:                                                     # only if at least one frame exists
        grid_img = build_grid(imgs)
        cv2.imwrite(str(EXAMPLES_DIR / f"grid_{fname}"), grid_img)

print("Done. Frames copied and grids saved to", EXAMPLES_DIR)
