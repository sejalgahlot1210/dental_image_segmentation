import os, cv2, torch, numpy as np
from torchvision import models
from torch import nn
from attention_unet import AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= MODELS =================
deeplab = models.segmentation.deeplabv3_resnet50(weights=None)
deeplab.classifier[4] = nn.Conv2d(256, 1, 1)
deeplab.load_state_dict(torch.load("dental_model.pth", map_location=device), strict=False)
deeplab.to(device).eval()

attunet = AttentionUNet()
attunet.load_state_dict(torch.load("attention_unet.pth", map_location=device))
attunet.to(device).eval()

# ================= TRANSFORM =================
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# ================= METRICS =================
def dice(p, g):
    p, g = p > 0, g > 0
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-6)

def iou(p, g):
    p, g = p > 0, g > 0
    return (p & g).sum() / ((p | g).sum() + 1e-6)

def pixel_accuracy(p, g):
    p, g = p > 0, g > 0
    return np.mean(p == g)

def precision(p, g):
    p, g = p > 0, g > 0
    tp = (p & g).sum()
    fp = (p & (~g)).sum()
    return tp / (tp + fp + 1e-6)

def recall(p, g):
    p, g = p > 0, g > 0
    tp = (p & g).sum()
    fn = ((~p) & g).sum()
    return tp / (tp + fn + 1e-6)

# ================= EVALUATION =================
D, I, PA, P, R = [], [], [], [], []

for f in os.listdir("masks"):
    if not f.endswith(".png"):
        continue

    img = cv2.imread("cropped_images/" + f.replace(".png", ".jpg"))
    gt = cv2.imread("masks/" + f, 0)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(image=rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(deeplab(x)["out"])
        p2 = torch.sigmoid(attunet(x))

    # Ensemble (average)
    p = ((p1 + p2) / 2)[0][0].cpu().numpy()
    p = cv2.resize(p, (gt.shape[1], gt.shape[0])) > 0.5

    D.append(dice(p, gt))
    I.append(iou(p, gt))
    PA.append(pixel_accuracy(p, gt))
    P.append(precision(p, gt))
    R.append(recall(p, gt))

# ================= RESULTS =================
print("======================================")
print("Ensemble Results (DeepLabV3 + Attention U-Net)")
print(f"Average Dice Score     : {np.mean(D):.4f}")
print(f"Average IoU Score      : {np.mean(I):.4f}")
print(f"Pixel Accuracy         : {np.mean(PA):.4f}")
print(f"Precision              : {np.mean(P):.4f}")
print(f"Recall                 : {np.mean(R):.4f}")
print("======================================")
