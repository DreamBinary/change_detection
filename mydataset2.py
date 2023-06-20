import os

import torch
from torch.utils.data import Dataset
import mmcv


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
        return torch.cat((img1, img2), 0), mask
