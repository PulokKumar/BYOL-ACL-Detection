import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image


class KneeInjuryDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]
        image_path = os.path.join(folder_path, 'slice_00.png')
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label


class KneeInjuryUnlabeledDataset(Dataset):
    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)

        return view1, view2


def byol_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.8),

        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def data_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def load_encoder_weights_into(encoder, ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    encoder_dict = encoder.state_dict()
    for k in list(state.keys()):
        if k in encoder_dict:
            encoder_dict[k] = state[k]
    encoder.load_state_dict(encoder_dict)
    print("Loaded encoder weights from", ckpt_path)
