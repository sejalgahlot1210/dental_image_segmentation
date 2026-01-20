import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

print("Script started...")

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= DATASET =================
class DentalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = []

        for img in os.listdir(img_dir):
            if not img.lower().endswith(".jpg"):
                continue

            mask_path = os.path.join(mask_dir, img.replace(".jpg", ".png"))
            if os.path.exists(mask_path):
                self.images.append(img)

        print(f"Total training pairs found: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise ValueError(f"Mask missing for {img_name}")

        mask = (mask > 127).astype("float32")

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        return image, mask.unsqueeze(0)

# ================= TRANSFORMS =================
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),
    ToTensorV2()
])

# ================= DATA LOADER =================
print("Loading dataset...")

dataset = DentalDataset("cropped_images", "masks", transform)

if len(dataset) == 0:
    raise RuntimeError("No image-mask pairs found. Check masks folder.")

loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

print("DataLoader ready.")

# ================= MODEL =================
print("Loading DeepLabV3+ model...")

model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
model = model.to(device)

print("Model loaded.")

# ================= LOSS =================
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        inter = (pred * target).sum()
        return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()

# ================= OPTIMIZER =================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAINING =================
epochs = 25
print("Starting training...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for imgs, masks in loop:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)["out"]
        loss = bce(outputs, masks) + dice(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} average loss: {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), "dental_model.pth")
print("Training finished. Model saved as dental_model.pth")
