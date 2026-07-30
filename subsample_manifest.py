"""
One-off script: caps maize at a max count in the manifest, so it doesn't
dominate training time. Keeps proportional train/val split assignment
already in the file.
"""
import csv
import random

MAX_MAIZE = 4000
random.seed(42)

with open("datasets/manifest.csv") as f:
    rows = list(csv.DictReader(f))

maize_rows = [r for r in rows if r["crop"] == "maize"]
other_rows = [r for r in rows if r["crop"] != "maize"]

random.shuffle(maize_rows)
maize_sample = maize_rows[:MAX_MAIZE]

final_rows = other_rows + maize_sample
random.shuffle(final_rows)

with open("datasets/manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "crop", "cause", "deficiency_class", "severity", "split"])
    writer.writeheader()
    writer.writerows(final_rows)

print(f"Reduced manifest: {len(final_rows)} rows (maize capped at {len(maize_sample)})")
