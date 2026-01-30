import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from attention_unet import AttentionUNet

# ================= CONFIG =================
IMAGE_DIR = "cropped_images"
OUTPUT_DIR = "batch_results/attention_unet"
MODEL_PATH = "attention_unet.pth"
IMG_SIZE = 256
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
model = AttentionUNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Attention U-Net loaded.")

# ================= TRANSFORM =================
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# ================= INFERENCE =================
for img_name in sorted(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        pred = (pred > THRESHOLD).float()

    mask = pred.squeeze().cpu().numpy() * 255
    mask = mask.astype(np.uint8)

    # Save raw mask
    mask_path = os.path.join(
        OUTPUT_DIR,
        img_name.replace(".jpg", ".png")
    )
    cv2.imwrite(mask_path, mask)

    # ---------- Overlay ----------
    orig_vis = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 1] = mask_color[:, :, 1]  # green

    overlay = cv2.addWeighted(orig_vis, 0.7, mask_color, 0.3, 0)

    overlay_path = os.path.join(
        OUTPUT_DIR,
        img_name.replace(".jpg", "_overlay.png")
    )
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("✅ Attention U-Net segmented results saved.")
