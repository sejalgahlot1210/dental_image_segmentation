import os, cv2, torch, numpy as np
from torchvision import models
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.segmentation.deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(256,1,1)
model.load_state_dict(torch.load("best_deeplabv3.pth", map_location=device), strict=False)
model.to(device)
model.eval()

transform = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

def metrics(p, g):
    p, g = p>0, g>0
    tp = (p & g).sum()
    fp = (p & (~g)).sum()
    fn = ((~p) & g).sum()
    dice = 2*tp/(p.sum()+g.sum()+1e-6)
    iou = tp/((p|g).sum()+1e-6)
    acc = (p==g).mean()
    prec = tp/(tp+fp+1e-6)
    rec = tp/(tp+fn+1e-6)
    return dice, iou, acc, prec, rec

D,I,A,P,R = [],[],[],[],[]

for f in os.listdir("dataset/masks/test"):
    if not f.endswith(".png"): continue

    img = cv2.cvtColor(cv2.imread("dataset/images/test/"+f.replace(".png",".jpg")), cv2.COLOR_BGR2RGB)
    gt = cv2.imread("dataset/masks/test/"+f,0)

    x = transform(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(x)["out"])[0][0].cpu().numpy()
    p = cv2.resize(p,(gt.shape[1],gt.shape[0]))>0.5

    d,i,a,pr,re = metrics(p,gt)
    D.append(d); I.append(i); A.append(a); P.append(pr); R.append(re)

print("DeepLabV3 Test Results")
print("Dice:",np.mean(D))
print("IoU:",np.mean(I))
print("Pixel Acc:",np.mean(A))
print("Precision:",np.mean(P))
print("Recall:",np.mean(R))
