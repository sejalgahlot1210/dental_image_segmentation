import os, cv2, torch, numpy as np
from attention_unet import AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= MODEL =================
model = AttentionUNet().to(device)
model.load_state_dict(torch.load("attention_unet.pth", map_location=device))
model.eval()

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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = transform(image=img)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        p = torch.sigmoid(model(x))[0][0].cpu().numpy()

    p = cv2.resize(p, (gt.shape[1], gt.shape[0])) > 0.5

    D.append(dice(p, gt))
    I.append(iou(p, gt))
    PA.append(pixel_accuracy(p, gt))
    P.append(precision(p, gt))
    R.append(recall(p, gt))

# ================= RESULTS =================
print("======================================")
print("Attention U-Net Results")
print(f"Average Dice Score     : {np.mean(D):.4f}")
print(f"Average IoU Score      : {np.mean(I):.4f}")
print(f"Pixel Accuracy         : {np.mean(PA):.4f}")
print(f"Precision              : {np.mean(P):.4f}")
print(f"Recall                 : {np.mean(R):.4f}")
print("======================================")
