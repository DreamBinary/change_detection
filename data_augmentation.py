import random
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF


class RandomErasing(transforms.RandomErasing):
    def forward(self, img1, img2, mask):
        if torch.rand(1) < self.p:
            if isinstance(self.value, (int, float)):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value
            if value is not None and not (len(value) in (1, img1.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img1.shape[-3]} (number of input channels)"
                )
            x, y, h, w, v = self.get_params(img1, scale=self.scale, ratio=self.ratio, value=value)
            return TF.erase(img1, x, y, h, w, v, self.inplace), TF.erase(img2, x, y, h, w, v, self.inplace), TF.erase(
                mask, x, y, h, w, v, self.inplace)
        return img1, img2, mask


class RandomAdjustSharpness(transforms.RandomSolarize):

    def forward(self, img1, img2, mask):
        if torch.rand(1).item() < self.p:
            return TF.adjust_sharpness(img1, self.sharpness_factor), TF.adjust_sharpness(img2,
                                                                                         self.sharpness_factor), mask
        return img1, img2, mask


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
