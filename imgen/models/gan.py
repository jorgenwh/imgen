import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.image_head = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
        )
        self.label_head = nn.Sequential(
            nn.Linear(10, 128),
            nn.LeakyReLU(0.2),
        )
        self.body = nn.Sequential(
            nn.Linear(128 * 7 * 7 + 128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        image_features = self.image_head(image)
        label_features = self.label_head(label)
        combined = torch.cat([image_features, label_features], dim=1)
        return self.body(combined)


class Generator(nn.Module):
    def __init__(self, noise_dim: int):
        super(Generator, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(noise_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),

            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def weights_init(m):
    """DCGAN weight initialization: N(0, 0.02)"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
