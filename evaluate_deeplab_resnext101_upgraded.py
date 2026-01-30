import os, cv2, torch, numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

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

# ================= METRICS =================
def dice_score(pred, gt):
    pred = (pred > 0.5).float()
    inter = (pred * gt).sum()
    return (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)

def iou_score(pred, gt):
    pred = (pred > 0.5).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)

def pixel_accuracy(pred, gt):
    pred = (pred > 0.5).float()
    correct = (pred == gt).sum()
    total = gt.numel()
    return correct / (total + 1e-6)

def precision_score(pred, gt):
    pred = (pred > 0.5).float()
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    return tp / (tp + fp + 1e-6)

def recall_score(pred, gt):
    pred = (pred > 0.5).float()
    tp = (pred * gt).sum()
    fn = ((1 - pred) * gt).sum()
    return tp / (tp + fn + 1e-6)

# ================= TRANSFORM =================
val_tf = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# ================= DATA =================
ds = DentalDataset("cropped_images", "masks", val_tf)
dl = DataLoader(ds, batch_size=1, shuffle=False)

# ================= MODEL =================
model = smp.DeepLabV3Plus(
    encoder_name="resnext101_32x8d",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    encoder_output_stride=8
).to(device)

model.load_state_dict(
    torch.load("deeplab_resnext101_upgraded.pth", map_location=device)
)
model.eval()

# ================= EVALUATION =================
dice_total, iou_total = 0, 0
pixel_total, prec_total, recall_total = 0, 0, 0

with torch.no_grad():
    for x, y in tqdm(dl, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x))

        dice_total   += dice_score(pred, y).item()
        iou_total    += iou_score(pred, y).item()
        pixel_total  += pixel_accuracy(pred, y).item()
        prec_total   += precision_score(pred, y).item()
        recall_total += recall_score(pred, y).item()

n = len(dl)

print("\n📊 FINAL RESULTS (Upgraded DeepLab ResNeXt101)")
print(f"Average Dice Score     : {dice_total / n:.4f}")
print(f"Average IoU Score      : {iou_total / n:.4f}")
print(f"Pixel Accuracy         : {pixel_total / n:.4f}")
print(f"Precision              : {prec_total / n:.4f}")
print(f"Recall                 : {recall_total / n:.4f}")
