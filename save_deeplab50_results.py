import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models

# ================= CONFIG =================
IMAGE_DIR = "cropped_images"                     # input images
OUTPUT_DIR = "batch_results/deeplab"     # output folder
MODEL_PATH = "dental_model.pth"          # trained model
IMG_SIZE = 256
THRESHOLD = 0.5

# Create output directory safely
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
model = models.segmentation.deeplabv3_resnet50(weights=None)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device),
    strict=False
)

model.to(device)
model.eval()

print("Model loaded.")

# ================= TRANSFORM =================
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# ================= INFERENCE LOOP =================
for img_name in sorted(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image)
    x = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)["out"]
        pred = torch.sigmoid(pred)
        pred = (pred > THRESHOLD).float()

    mask = pred.squeeze().cpu().numpy() * 255
    mask = mask.astype(np.uint8)

    save_path = os.path.join(
        OUTPUT_DIR,
        img_name.replace(".jpg", ".png")
    )

    cv2.imwrite(save_path, mask)

    # Resize original image to match mask (important)
    orig_vis = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert mask to 3-channel
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Color the mask (green teeth)
    mask_color[:, :, 1] = mask_color[:, :, 1]  # green channel
    mask_color[:, :, 0] = 0                    # remove blue
    mask_color[:, :, 2] = 0                    # remove red

    # Overlay mask on original image
    overlay = cv2.addWeighted(orig_vis, 0.7, mask_color, 0.3, 0)

    # Save overlay image
    overlay_path = os.path.join(
        OUTPUT_DIR,
        img_name.replace(".jpg", "_overlay.png")
    )

    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"✅ Segmented results saved in: {OUTPUT_DIR}")
