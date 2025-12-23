import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CharTokenizer:
    """Character-level tokenizer. Handles any text input."""

    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        self.chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}  # 0 = pad
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.idx_to_char[0] = ""
        self.vocab_size = len(self.chars) + 1  # +1 for pad

    def encode(self, text: str) -> list[int]:
        tokens = [self.char_to_idx.get(c, 0) for c in text[: self.max_len]]
        tokens += [0] * (self.max_len - len(tokens))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.idx_to_char.get(t, "") for t in tokens)


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
    """
    MS-COCO Captions dataset via torchvision.

    Requires manual download:
        - Images: http://images.cocodataset.org/zips/train2017.zip
        - Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    Expected structure:
        data_dir/
            train2017/
                000000000001.jpg
                ...
            annotations/
                captions_train2017.json
    """

    def __init__(self, data_dir: str, img_size: int, tokenizer: CharTokenizer, split: str = "train"):
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        img_dir = f"{data_dir}/{split}2017"
        ann_file = f"{data_dir}/annotations/captions_{split}2017.json"

        self.dataset = datasets.CocoCaptions(
            root=img_dir,
            annFile=ann_file,
            transform=self.transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, captions = self.dataset[idx]
        caption = captions[0]  # Use first caption
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
        return image, tokens


def get_coco_dataloader(
    batch_size: int,
    img_size: int,
    tokenizer: CharTokenizer,
    data_dir: str = "./data/coco",
    split: str = "train",
) -> DataLoader:
    dataset = COCODataset(data_dir=data_dir, img_size=img_size, tokenizer=tokenizer, split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return dataloader
