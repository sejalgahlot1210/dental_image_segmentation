import os, cv2, torch, numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ================= DEVICE =================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ================= DATASET =================
class DentalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.imgs = [f.replace(".png", ".jpg") for f in os.listdir(mask_dir)]
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.img_dir, self.imgs[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(self.mask_dir, self.imgs[i].replace(".jpg", ".png")), 0
        )
        mask = (mask > 127).astype("float32")

        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)

    def __len__(self):
        return len(self.imgs)

# ================= TRANSFORMS =================
train_tf = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),
    ToTensorV2()
])

# ================= DATA =================
ds = DentalDataset("cropped_images", "masks", train_tf)
dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)

# ================= MODEL (UPGRADED) =================
model = smp.DeepLabV3Plus(
    encoder_name="resnext101_32x8d",
    encoder_weights="imagenet",      
    in_channels=3,
    classes=1,
    encoder_output_stride=8          
).to(device)

# ================= LOSSES =================
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = smp.losses.SoftBCEWithLogitsLoss(
    pos_weight=torch.tensor(2.0).to(device)
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAINING =================
EPOCHS = 25

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, y in loop:
        x, y = x.to(device), y.to(device)

        preds = model(x)
        loss = dice_loss(preds, y) + bce_loss(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

# ================= SAVE =================
torch.save(model.state_dict(), "deeplab_resnext101_upgraded.pth")
print("✅ Upgraded DeepLab ResNeXt101 training completed.")
