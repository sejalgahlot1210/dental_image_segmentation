import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= Device =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= Load Model =================
model = smp.DeepLabV3Plus(
    encoder_name="resnext101_32x8d",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    encoder_output_stride=16
).to(device)

model.load_state_dict(torch.load("deeplab_inception.pth", map_location=device))
model.eval()

# ================= Transform =================
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# ================= Metrics =================
def dice_score(p, g):
    p, g = p > 0, g > 0
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-6)

def iou_score(p, g):
    p, g = p > 0, g > 0
    return (p & g).sum() / ((p | g).sum() + 1e-6)

def pixel_accuracy(p, g):
    p, g = p > 0, g > 0
    return np.mean(p == g)

def precision_score(p, g):
    p, g = p > 0, g > 0
    tp = (p & g).sum()
    fp = (p & (~g)).sum()
    return tp / (tp + fp + 1e-6)

def recall_score(p, g):
    p, g = p > 0, g > 0
    tp = (p & g).sum()
    fn = ((~p) & g).sum()
    return tp / (tp + fn + 1e-6)

# ================= Evaluation =================
D, I, PA, P, R = [], [], [], [], []

for mask_name in os.listdir("masks"):
    if not mask_name.endswith(".png"):
        continue

    img_name = mask_name.replace(".png", ".jpg")

    img = cv2.imread("cropped_images/" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt = cv2.imread("masks/" + mask_name, 0)

    x = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0][0].cpu().numpy()

    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0])) > 0.5

    D.append(dice_score(pred, gt))
    I.append(iou_score(pred, gt))
    PA.append(pixel_accuracy(pred, gt))
    P.append(precision_score(pred, gt))
    R.append(recall_score(pred, gt))

print("===================================")
print("DeepLab ResNeXt101 Results")
print(f"Average Dice Score     : {np.mean(D):.4f}")
print(f"Average IoU Score      : {np.mean(I):.4f}")
print(f"Pixel Accuracy         : {np.mean(PA):.4f}")
print(f"Precision              : {np.mean(P):.4f}")
print(f"Recall                 : {np.mean(R):.4f}")
print("===================================")
