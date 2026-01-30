import os, cv2, torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

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

        mask = cv2.imread(os.path.join(self.mask_dir, img.replace(".jpg",".png")), 0)
        mask = (mask > 127).astype("float32")

        aug = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)

# ================= TRANSFORMS =================
train_tf = A.Compose([
    A.Resize(256,256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

# ================= LOAD DATA =================
train_ds = DentalDataset("dataset/images/train", "dataset/masks/train", train_tf)
val_ds   = DentalDataset("dataset/images/val", "dataset/masks/val", val_tf)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1)

# ================= MODEL =================
model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(256, 1, 1)
model = model.to(device)

# ================= LOSS =================
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        return 1 - (2*inter + 1e-6)/(pred.sum()+target.sum()+1e-6)

bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= METRIC =================
def dice_score(p, g):
    p = (p > 0.5).float()
    return (2*(p*g).sum())/(p.sum()+g.sum()+1e-6)

# ================= TRAINING =================
best_dice = 0
EPOCHS = 25

for epoch in range(EPOCHS):
    model.train()
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)["out"]
        loss = bce(out, masks) + dice(out, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs)["out"])
            dices.append(dice_score(preds, masks).item())

    val_dice = np.mean(dices)
    print("Val Dice:", val_dice)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "best_deeplabv3.pth")
        print("🔥 Best DeepLabV3 saved")

print("Training finished")
