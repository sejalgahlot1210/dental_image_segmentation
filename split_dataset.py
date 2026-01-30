#splits dataset into train-test set

import os
import random
import shutil

IMAGE_DIR = "cropped_images"
MASK_DIR = "masks"

OUT_IMG = "dataset/images"
OUT_MASK = "dataset/masks"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_IMG, split), exist_ok=True)
    os.makedirs(os.path.join(OUT_MASK, split), exist_ok=True)

# Collect valid pairs
images = []
for img in os.listdir(IMAGE_DIR):
    if not img.lower().endswith(".jpg"):
        continue
    mask = img.replace(".jpg", ".png")
    if os.path.exists(os.path.join(MASK_DIR, mask)):
        images.append(img)

random.shuffle(images)

n = len(images)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_imgs = images[:n_train]
val_imgs = images[n_train:n_train + n_val]
test_imgs = images[n_train + n_val:]

def copy_files(file_list, split):
    for img in file_list:
        mask = img.replace(".jpg", ".png")

        shutil.copy(os.path.join(IMAGE_DIR, img), os.path.join(OUT_IMG, split, img))
        shutil.copy(os.path.join(MASK_DIR, mask), os.path.join(OUT_MASK, split, mask))

copy_files(train_imgs, "train")
copy_files(val_imgs, "val")
copy_files(test_imgs, "test")

print("======================================")
print(f"Total images: {n}")
print(f"Train: {len(train_imgs)}")
print(f"Validation: {len(val_imgs)}")
print(f"Test: {len(test_imgs)}")
print("Dataset split completed successfully")
print("======================================")
