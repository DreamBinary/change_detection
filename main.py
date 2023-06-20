import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
import os
from torch.utils.data import Dataset
import mmcv
import argparse
import openbayestool
import random
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img1, img2, mask):
        if torch.rand(1) < self.p:
            return TF.hflip(img1), TF.hflip(img2), TF.hflip(mask)
        return img1, img2, mask


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def forward(self, img1, img2, mask):
        if torch.rand(1) < self.p:
            return TF.vflip(img1), TF.vflip(img2), TF.vflip(mask)
        return img1, img2, mask


class RandomRotation(torch.nn.Module):
    def __init__(self, degrees) -> None:
        super(RandomRotation, self).__init__()
        self.degrees = degrees

    def forward(self, img1, img2, mask):
        angle = random.choice(self.degrees)
        return TF.rotate(img1, angle), TF.rotate(img2, angle), TF.rotate(mask, angle)


class RandomCrop(transforms.RandomCrop):
    def forward(self, img1, img2, mask):
        h, w = 256, 256
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return TF.crop(img1, i, j, th, tw), TF.crop(img2, i, j, th, tw), TF.crop(mask, i, j, th, tw),


class Compose(transforms.Compose):
    def __call__(self, img1, img2, mask):
        for t in self.transforms:
            img1, img2, mask = t(img1, img2, mask)
        return img1, img2, mask


class ToTensor(transforms.ToTensor):
    def __call__(self, img1, img2, mask):
        return TF.to_tensor(img1), TF.to_tensor(img2), TF.to_tensor(mask)


class MyDataset(Dataset):

    def __init__(self, root, is_train=True, transform=None, normalize=None) -> None:
        super().__init__()

        self.img1_dir = os.path.join(root, "train" if is_train else "val", "time1")
        self.img2_dir = os.path.join(root, "train" if is_train else "val", "time2")
        self.mask_dir = os.path.join(root, "train" if is_train else "val", "label")
        self.ann_file = os.path.join(root, "train.txt" if is_train else "val.txt")
        self.imgs = self.get_imgs()
        self.transform = transform
        self.normalize = normalize

    def get_imgs(self):
        imgs = []
        with open(self.ann_file) as f:
            for i in f:
                i = i.strip()
                img = dict(
                    img1=os.path.join(self.img1_dir, i),
                    img2=os.path.join(self.img2_dir, i),
                    mask=os.path.join(self.mask_dir, i)
                )
                imgs.append(img)
        return imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img1 = mmcv.imread(self.imgs[idx]["img1"], channel_order="rgb")
        img2 = mmcv.imread(self.imgs[idx]["img2"], channel_order="rgb")
        mask = mmcv.imread(self.imgs[idx]["mask"], flag="grayscale")
        mask = mask / 255.0

        if self.transform is not None:
            img1, img2, mask = self.transform(img1, img2, mask)
        if self.normalize is not None:
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
        return img1, img2, mask


parser = argparse.ArgumentParser(description="hypertuning")
parser.add_argument("--input", help="input")
parser.add_argument("--lr", help="lr")
args = parser.parse_args()

lr = float(args.lr)

train_set = MyDataset(root=args.input, is_train=True, transform=Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation([0, 90, 180, 270]),
]), normalize=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

val_set = MyDataset(root=args.input, is_train=False, transform=Compose([ToTensor()]),
                    normalize=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

train_loader = DataLoader(
    train_set,
    batch_size=100,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_set,
    batch_size=100,
    shuffle=False,
    num_workers=0
)

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=6,
    classes=2,
)
model.cuda()

criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optim, mode='max', factor=0.1, patience=2,
#     verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=True)


def train_model(epoch):
    model.train()
    print(f"Epoch {epoch} Training")
    with tqdm(train_loader, desc=str(epoch)) as it:
        for idx, (img1, img2, mask) in enumerate(it, 0):
            img1, img2, mask = img1.cuda(), img2.cuda(), mask.cuda()
            optim.zero_grad()
            mask = mask.long()
            with autocast():
                img = torch.cat((img1, img2), 1)
                outputs = model(img)
                mask = mask.squeeze(1)
                loss = criterion(outputs, mask)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            _, pred = torch.max(outputs.data, 1)
            p, r, f1, iou = get_index(pred, mask)
            it.set_postfix_str(f"loss: {loss.item(): .4f} p: {p: .4f}  r: {r: .4f}  f1: {f1: .4f}  iou: {iou: .4f}")


def get_index(pred, label):
    eps = 1e-7
    tp = torch.sum(label * pred)
    fp = torch.sum(pred) - tp
    fn = torch.sum(label) - tp

    p = (tp + eps) / (tp + fp + eps)
    r = (tp + eps) / (tp + fn + eps)
    f1 = (2 * p * r + eps) / (p + r + eps)
    iou = (tp + eps) / (tp + fn + fp + eps)
    return p, r, f1, iou


def test_model(epoch):
    model.eval()
    global max_score
    f1s = 0
    print(f"Epoch {epoch} Testing")
    with torch.no_grad():
        with tqdm(val_loader, desc=str(epoch)) as it:
            for img1, img2, mask in it:
                img1, img2, mask = img1.cuda(), img2.cuda(), mask.cuda()
                img = torch.cat((img1, img2), 1)
                outputs = model(img)
                _, pred = torch.max(outputs.data, 1)
                mask = mask.squeeze(1)
                p, r, f1, iou = get_index(pred, mask)
                f1s += f1
                it.set_postfix_str(f"p: {p: .4f}  r: {r: .4f}  f1: {f1: .4f}  iou: {iou: .4f}")
    f1s /= len(val_loader)

    openbayestool.log_metric("f1", f1s.item())

    # scheduler.step(f1s)
    print("f1", f1s.item())
    if max_score < f1s:
        max_score = f1s
        print('max_score', max_score.item())


num_epoch = 10
max_score = 0
for epoch in range(0, 10):
    train_model(epoch=epoch)
    test_model(epoch=epoch)
print("completed!")
print('max_score', max_score.item())
openbayestool.log_metric("f1", max_score.item())
# data_binding:
# - data: openbayes/CBmTHOI6btI/1
# path: /input0
# resource: vgpu
# env: pytorch-1.8
# command: "python main.py --input=/input0"
# hyper_tuning:
# max_job_count: 5
# hyperparameter_metric: max_score
# side_metrics: ["f1"]
# goal: MAXIMIZE
# algorithm: Grid
# parameter_specs:
# - name: 'lr'
# type: DOUBLE
# min_value: 0.00001
# max_value: 0.001
# scale_type: UNIT_LINEAR_SCALE
