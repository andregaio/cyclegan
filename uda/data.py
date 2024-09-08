import os
import random

import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class UDADataset(Dataset):
    def __init__(self, dataset_dir, max_iterations=1334):

        self.dataset_dir = dataset_dir
        self.max_iterations = max_iterations

        src_path = f"{self.dataset_dir}/source"
        tgt_path = f"{self.dataset_dir}/target"

        self.source_imgs = [f"{src_path}/{img}" for img in os.listdir(src_path)]
        self.target_imgs = [f"{tgt_path}/{img}" for img in os.listdir(tgt_path)]

    def __len__(self):
        return self.max_iterations

    def __getitem__(self, idx):

        source_img = Image.open(random.choice(self.source_imgs)).convert("RGB")
        target_img = Image.open(random.choice(self.target_imgs)).convert("RGB")

        source_tensor = image_transforms(source_img)
        target_tensor = image_transforms(target_img)

        return source_tensor, target_tensor


def image_transforms(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.ConvertImageDtype(torch.float),  # Convert to float
            transforms.Resize(int(256 * 1.15), interpolation=Image.BICUBIC),  # Resize
            transforms.RandomCrop(256),  # Randomly crop to 256x256
            transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(image)


def debug_transforms(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.ConvertImageDtype(torch.float),  # Convert to float
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(image)
