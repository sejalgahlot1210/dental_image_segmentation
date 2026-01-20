import os
import cv2
import torch
import numpy as np
from torchvision import models
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= LOAD MODEL =================
model = models.segmentation.deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

state_dict = torch.load("dental_model.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

# ================= TRANSFORM =================
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# ================= METRICS =================
def dice_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-6)

def iou_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + 1e-6)

image_dir = "cropped_images"
mask_dir = "masks"

dice_scores = []
iou_scores = []

for mask_name in os.listdir(mask_dir):

    if not mask_name.endswith(".png"):
        continue

    img_name = mask_name.replace(".png", ".jpg")

    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_mask = cv2.imread(mask_path, 0)

    aug = transform(image=image)
    img_tensor = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)["out"]
        pred = torch.sigmoid(output)[0][0].cpu().numpy()

    pred = cv2.resize(pred, (gt_mask.shape[1], gt_mask.shape[0]))
    pred = pred > 0.5

    dice_scores.append(dice_score(pred, gt_mask))
    iou_scores.append(iou_score(pred, gt_mask))

print("======================================")
print("Total test images:", len(dice_scores))
print("Average Dice Score:", np.mean(dice_scores))
print("Average IoU Score :", np.mean(iou_scores))
print("======================================")
