#deep lab using inception instead of resnet
import os, cv2, torch, numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DentalDataset(Dataset):
    def __init__(self,img_dir,mask_dir,transform):
        self.imgs = [f.replace(".png",".jpg") for f in os.listdir(mask_dir)]
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __getitem__(self,i):
        img = cv2.imread(self.img_dir+"/"+self.imgs[i])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_dir+"/"+self.imgs[i].replace(".jpg",".png"),0)
        mask = (mask>127).astype("float32")
        aug = self.transform(image=img,mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)

    def __len__(self):
        return len(self.imgs)

transform = A.Compose([
    A.Resize(256,256),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(),
    ToTensorV2()
])

ds = DentalDataset("cropped_images","masks",transform)
dl = DataLoader(ds,batch_size=2,shuffle=True,drop_last=True)

model = smp.DeepLabV3Plus(
    encoder_name="resnext101_32x8d",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    encoder_output_stride=16
).to(device)


dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss  = smp.losses.SoftBCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(),1e-4)

for e in range(25):
    loop = tqdm(dl)
    for x,y in loop:
        x,y = x.to(device),y.to(device)
        p = model(x)
        l = dice_loss(p, y) + bce_loss(p, y)
        opt.zero_grad()
        l.backward()
        opt.step()
        loop.set_description(f"Epoch {e+1}")
        loop.set_postfix(loss=l.item())

torch.save(model.state_dict(),"deeplab_inception.pth")
print("DeepLab+Inception trained")