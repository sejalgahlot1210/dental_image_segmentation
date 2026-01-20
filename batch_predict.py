import os
import cv2
import torch
import numpy as np
from torchvision import models
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= LOAD MODEL =================
model = models.segmentation.deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

state_dict = torch.load("dental_model.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

print("Model loaded successfully.")

# ================= TRANSFORM =================
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

image_dir = "cropped_images"
output_dir = "batch_results"

os.makedirs(output_dir, exist_ok=True)

# ================= BATCH PREDICTION =================
for img_name in os.listdir(image_dir):

    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, img_name)

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aug = transform(image=image_rgb)
    img_tensor = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)["out"]
        pred = torch.sigmoid(output)[0][0].cpu().numpy()

    # Resize prediction to original image size
    pred = cv2.resize(pred, (image.shape[1], image.shape[0]))
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Create green overlay
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = pred_mask

    overlay = cv2.addWeighted(image, 0.7, green_mask, 0.3, 0)

    save_path = os.path.join(output_dir, img_name.replace(".jpg", "_overlay.png"))
    cv2.imwrite(save_path, overlay)

print("Batch prediction completed. Results saved in batch_results/")
