import os
import cv2
import torch
import numpy as np
from torchvision import models
from torch import nn
from attention_unet import AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ===== Load DeepLab =====
deeplab = models.segmentation.deeplabv3_resnet50(weights=None)
deeplab.classifier[4] = nn.Conv2d(256,1,1)
deeplab.load_state_dict(torch.load("dental_model.pth"), strict=False)
deeplab.to(device).eval()

# ===== Load Attention UNet =====
attunet = AttentionUNet()
attunet.load_state_dict(torch.load("attention_unet.pth"))
attunet.to(device).eval()

transform = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

os.makedirs("ensemble_results", exist_ok=True)

for img_name in os.listdir("cropped_images"):
    if not img_name.endswith(".jpg"):
        continue

    img = cv2.imread("cropped_images/"+img_name)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x = transform(image=rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(deeplab(x)["out"])
        p2 = torch.sigmoid(attunet(x))

    p = ((p1 + p2) / 2)[0][0].cpu().numpy()

    p = cv2.resize(p,(img.shape[1],img.shape[0]))
    mask = (p>0.5).astype(np.uint8)*255

    overlay = img.copy()
    overlay[:,:,1] = np.maximum(overlay[:,:,1], mask)

    cv2.imwrite("ensemble_results/"+img_name.replace(".jpg","_ensemble.png"), overlay)

print("Ensemble predictions saved in ensemble_results/")
