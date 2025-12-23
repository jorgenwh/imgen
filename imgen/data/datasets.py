import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from datasets import load_dataset
from PIL import Image


def get_mnist_dataloader(batch_size: int, data_dir: str = "./data") -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),                 # Converts to [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # Shifts to [-1, 1]
    ])

    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader


class COCODataset(Dataset):
    def __init__(self, split: str, img_size: int):
        self.dataset = load_dataset("nlphuji/flickr30k", split=split)  # Flickr30k is smaller, easier to start
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # RGB to [-1, 1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)

        # Get first caption (each image has multiple)
        caption = item["caption"][0] if isinstance(item["caption"], list) else item["caption"]

        return image, caption


def get_coco_dataloader(batch_size: int, img_size: int = 64, split: str = "train") -> DataLoader:
    dataset = COCODataset(split=split, img_size=img_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader
