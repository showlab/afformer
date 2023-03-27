from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F

class RandomSizeCrop(RandomResizedCrop):
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.crop(img, i, j, h, w)

if __name__ == '__main__':
    import torch
    x = torch.rand(1,3,256,256)
    f = RandomSizeCrop(-1, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
    print(f(x).shape)