"""
One-off script: reorganizes the Roboflow PlantDoc-Tomatoes multiclass
export into class-named folders. Unlike the deficiency export, this
one's _classes.csv only has Healthy/Unhealthy (too coarse and
inconsistent — some rows have both flags set), so classes are inferred
from filename prefixes instead, which are consistent and descriptive.

Run once: python reorganize_plantdoc.py
"""

import os
import re
import shutil

ROOT = "datasets/raw/tomato_plantdoc"
SPLITS = ["train", "valid", "test"]

# Ordered so longer/more-specific prefixes are checked before the bare
# "Tomato-leaf-" fallback (healthy) — first match wins.
PREFIX_TO_CLASS = [
    ("Tomato-Early-blight-leaf",     "Early_Blight"),
    ("Tomato-Septoria-leaf-spot",    "Septoria_Leaf_Spot"),
    ("Tomato-leaf-bacterial-spot",   "Bacterial_Spot"),
    ("Tomato-leaf-late-blight",      "Late_Blight"),
    ("Tomato-leaf-mosaic-virus",     "Mosaic_Virus"),
    ("Tomato-leaf-yellow-virus",     "Yellow_Leaf_Curl_Virus"),
    ("Tomato-mold-leaf",             "Leaf_Mold"),
    ("Tomato-leaf",                  "Healthy"),  # fallback: bare "Tomato-leaf-N-..."
]

for split in SPLITS:
    split_dir = os.path.join(ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"Skipping {split}: directory not found")
        continue

    moved, unmatched = 0, 0
    for fn in os.listdir(split_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        matched_class = None
        for prefix, class_name in PREFIX_TO_CLASS:
            if fn.startswith(prefix):
                matched_class = class_name
                break

        if matched_class is None:
            unmatched += 1
            continue

        class_folder = os.path.join(split_dir, matched_class)
        os.makedirs(class_folder, exist_ok=True)
        shutil.move(os.path.join(split_dir, fn), os.path.join(class_folder, fn))
        moved += 1

    print(f"{split}: moved {moved} images into class folders, "
          f"{unmatched} unmatched filenames")
