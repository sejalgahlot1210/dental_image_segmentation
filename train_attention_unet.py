import os, cv2, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from attention_unet import AttentionUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DentalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.imgs = []
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        for f in os.listdir(mask_dir):
            self.imgs.append(f.replace(".png",".jpg"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, self.imgs[idx].replace(".jpg",".png")),0)
        mask = (mask>127).astype("float32")

        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)

transform = A.Compose([
    A.Resize(256,256),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2()
])

dataset = DentalDataset("cropped_images","masks",transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

model = AttentionUNet().to(device)

class DiceLoss(nn.Module):
    def forward(self,p,t):
        p = torch.sigmoid(p)
        return 1 - (2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)

bce = nn.BCEWithLogitsLoss()
dice = DiceLoss()
opt = torch.optim.Adam(model.parameters(),1e-4)

for epoch in range(25):
    loop = tqdm(loader)
    for x,y in loop:
        x,y = x.to(device), y.to(device)
        p = model(x)
        loss = bce(p,y)+dice(p,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(),"attention_unet.pth")
print("Attention U-Net trained & saved")
