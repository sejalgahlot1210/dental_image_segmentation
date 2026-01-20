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

# ================= IMAGE TRANSFORM =================
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# ================= LOAD TEST IMAGE =================
img_path = "cropped_images/IMG_0456.jpg"   # change to any test image
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

aug = transform(image=image)
img_tensor = aug["image"].unsqueeze(0).to(device)

# ================= PREDICT =================
with torch.no_grad():
    output = model(img_tensor)["out"]
    mask = torch.sigmoid(output)[0][0].cpu().numpy()

# ================= SAVE RESULT =================
mask = (mask > 0.5).astype(np.uint8) * 255

cv2.imwrite("predicted_mask.png", mask)
print("Prediction saved as predicted_mask.png")
