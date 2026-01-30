#performing attention unet of split architecture

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= MODEL =================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, g_ch, x_ch, int_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, int_ch, 1),
            nn.BatchNorm2d(int_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, int_ch, 1),
            nn.BatchNorm2d(int_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_ch, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        psi = self.psi(F.relu(self.W_g(g) + self.W_x(x)))
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 64)
        self.c2 = ConvBlock(64, 128)
        self.c3 = ConvBlock(128, 256)
        self.c4 = ConvBlock(256, 512)
        self.c5 = ConvBlock(512, 1024)

        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.a4 = AttentionBlock(512, 512, 256)
        self.c6 = ConvBlock(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.a3 = AttentionBlock(256, 256, 128)
        self.c7 = ConvBlock(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.a2 = AttentionBlock(128, 128, 64)
        self.c8 = ConvBlock(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.a1 = AttentionBlock(64, 64, 32)
        self.c9 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(F.max_pool2d(c1, 2))
        c3 = self.c3(F.max_pool2d(c2, 2))
        c4 = self.c4(F.max_pool2d(c3, 2))
        c5 = self.c5(F.max_pool2d(c4, 2))

        u4 = self.u4(c5)
        c6 = self.c6(torch.cat([u4, self.a4(u4, c4)], dim=1))

        u3 = self.u3(c6)
        c7 = self.c7(torch.cat([u3, self.a3(u3, c3)], dim=1))

        u2 = self.u2(c7)
        c8 = self.c8(torch.cat([u2, self.a2(u2, c2)], dim=1))

        u1 = self.u1(c8)
        c9 = self.c9(torch.cat([u1, self.a1(u1, c1)], dim=1))

        return self.out(c9)

# ================= DATASET =================
class DentalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        image = cv2.imread(os.path.join(self.img_dir, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img.replace(".jpg", ".png")), 0)
        mask = (mask > 127).astype("float32")

        aug = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)

# ================= TRANSFORMS =================
train_tf = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.3),
    A.Rotate(limit=5, p=0.3),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# ================= LOAD DATA =================
train_ds = DentalDataset("dataset/images/train", "dataset/masks/train", train_tf)
val_ds   = DentalDataset("dataset/images/val", "dataset/masks/val", val_tf)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

# ================= TRAINING =================
model = AttentionUNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def dice_score(p, g):
    p = (p > 0.5).float()
    return (2 * (p * g).sum()) / (p.sum() + g.sum() + 1e-6)

best_dice = 0

for epoch in range(30):
    model.train()
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        loss = criterion(out, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs))
            dices.append(dice_score(preds, masks).item())

    val_dice = np.mean(dices)
    print("Val Dice:", val_dice)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "best_attention_unet.pth")
        print("🔥 Best model saved")

print("Training finished")
