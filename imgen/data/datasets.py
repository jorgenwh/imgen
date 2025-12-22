from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(batch_size, data_dir: str = "./data") -> DataLoader:
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
        drop_last=True,                        # Drop incomplete last batch for consistent batch sizes
    )

    return dataloader
