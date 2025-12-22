import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_channels: int, width: int, height: int):
        super(Discriminator, self).__init__()

        # Each conv with stride=2, kernel=4, padding=1 halves the spatial size
        final_width = width // 4
        final_height = height // 4

        self.main = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * final_width * final_height, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, noise_dim: int, num_channels: int, width: int, height: int):
        super(Generator, self).__init__()

        # Start at 1/4 of target size, upsample twice with stride=2
        self.start_width = width // 4
        self.start_height = height // 4

        self.main = nn.Sequential(
            nn.Linear(noise_dim, 256 * self.start_width * self.start_height),
            nn.ReLU(),
            nn.Unflatten(1, (256, self.start_height, self.start_width)),

            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def weights_init(m):
    """DCGAN weight initialization: N(0, 0.02)"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
