import torch
from PIL import Image
import os
import random
from torchvision import transforms

def compute_train_transform(seed=123456):
    random.seed(seed)
    torch.random.manual_seed(seed)

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([color_jitter], 0.8),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    return train_transform

def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])
    return test_transform


class GesDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None):
        imgs = []
        for fn in os.listdir(img_path):
            fn = fn.strip()
            imgs.append((fn, int(fn.split('.')[0].split('_')[1])))
        self.imgs = imgs
        self.img_path = img_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(self.img_path + fn).convert('RGB')

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return x_i, x_j, label