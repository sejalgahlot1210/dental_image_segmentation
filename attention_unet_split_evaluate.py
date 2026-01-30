#Evaluate on test set

import os
import cv2
import torch
import numpy as np
from attention_unet_split import AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AttentionUNet().to(device)
model.load_state_dict(torch.load("best_attention_unet.pth", map_location=device))
model.eval()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

def dice(p, g):
    p, g = p > 0, g > 0
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-6)

def iou(p, g):
    p, g = p > 0, g > 0
    return (p & g).sum() / ((p | g).sum() + 1e-6)

D, I = [], []

for f in os.listdir("dataset/masks/test"):
    if not f.endswith(".png"):
        continue

    img = cv2.imread("dataset/images/test/" + f.replace(".png", ".jpg"))
    gt = cv2.imread("dataset/masks/test/" + f, 0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        p = torch.sigmoid(model(x))[0][0].cpu().numpy()

    p = cv2.resize(p, (gt.shape[1], gt.shape[0])) > 0.5

    D.append(dice(p, gt))
    I.append(iou(p, gt))

print("======================================")
print("Attention U-Net (Test Set)")
print("Average Dice:", np.mean(D))
print("Average IoU :", np.mean(I))
print("======================================")
