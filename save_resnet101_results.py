import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= CONFIG =================
IMAGE_DIR = "cropped_images"
OUTPUT_DIR = "batch_results/deeplab_resnext101"
MODEL_PATH = "deeplab_inception.pth"   # ← your saved file
IMG_SIZE = 256
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
model = smp.DeepLabV3Plus(
    encoder_name="resnext101_32x8d",
    encoder_weights=None,          # IMPORTANT: None for inference
    in_channels=3,
    classes=1,
    encoder_output_stride=16
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("DeepLab ResNeXt101 model loaded.")

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
        pred = model(x)                 # ← SMP returns tensor directly
        pred = torch.sigmoid(pred)
        pred = (pred > THRESHOLD).float()

    mask = pred.squeeze().cpu().numpy() * 255
    mask = mask.astype(np.uint8)

    # ---------- Save raw mask ----------
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, img_name.replace(".jpg", ".png")),
        mask
    )

    # ---------- Save overlay ----------
    orig_vis = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 1] = mask_color[:, :, 1]  # green channel

    overlay = cv2.addWeighted(orig_vis, 0.7, mask_color, 0.3, 0)

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, img_name.replace(".jpg", "_overlay.png")),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

print("✅ DeepLab ResNeXt101 segmented results saved.")
