import os
import json
import cv2
import numpy as np

JSON_DIR = "cropped_images"
MASK_DIR = "masks"

os.makedirs(MASK_DIR, exist_ok=True)

for file in os.listdir(JSON_DIR):
    if not file.endswith(".json"):
        continue

    with open(os.path.join(JSON_DIR, file)) as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        if shape["label"].lower() == "teeth":
            pts = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    mask_name = file.replace(".json", ".png")
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)

print("✅ Masks created successfully")
