"""
One-off script: reorganizes the Roboflow tomato multiclass export
(flat images + _classes.csv per split) into class-named folders,
matching the pattern dataset.py's RAW_LABEL_MAP expects.

Run once: python reorganize_tomato.py
"""

import csv
import os
import shutil

ROOT = "datasets/raw/tomato_deficiency_roboflow"
SPLITS = ["train", "valid", "test"]

# CSV column name -> folder name we'll use (matches RAW_LABEL_MAP style)
CLASS_MAP = {
    "Healthy": "Healthy",
    "Iron Deficiency": "Iron_Deficiency",
    "Magnesium Deficiency": "Magnesium_Deficiency",
    "Manganese Deficiency": "Manganese_Deficiency",
    "Nitrogen Deficiency": "Nitrogen_Deficiency",
    "Phosphorus Deficiency": "Phosphorus_Deficiency",
    "Potassium Deficiency": "Potassium_Deficiency",
}

for split in SPLITS:
    split_dir = os.path.join(ROOT, split)
    csv_path = os.path.join(split_dir, "_classes.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {split}: no _classes.csv found")
        continue

    moved, multi_label_skipped = 0, 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            # Find which class column(s) are marked 1
            active = [CLASS_MAP[col] for col in CLASS_MAP if row.get(col, "0").strip() == "1"]

            if len(active) != 1:
                # Skip images with 0 or 2+ labels active (ambiguous/multi-label
                # edge cases) rather than guessing — keeps single-label clean.
                multi_label_skipped += 1
                continue

            class_folder = os.path.join(split_dir, active[0])
            os.makedirs(class_folder, exist_ok=True)

            src = os.path.join(split_dir, filename)
            dst = os.path.join(class_folder, filename)
            if os.path.exists(src):
                shutil.move(src, dst)
                moved += 1

    print(f"{split}: moved {moved} images into class folders, "
          f"skipped {multi_label_skipped} ambiguous/multi-label images")
