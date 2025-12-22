import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from imgen.data.datasets import get_mnist_dataloader
from imgen.models.gan import Discriminator, Generator, weights_init


def train_gan(
        noise_dim: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        device: torch.device,
        output_dir: str
):
    dataloader = get_mnist_dataloader(batch_size=batch_size, data_dir="./data")

    discriminator = Discriminator().to(device)
    generator = Generator(noise_dim + 10).to(device)

    # Apply DCGAN weight initialization
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1) + "/" + str(epochs))
        bar = tqdm(dataloader, desc="Training", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} - {unit}")
        for real_images, real_labels in bar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)

            # Train Discriminator

            # 1. Real images with their TRUE labels -> should be 1
            real_onehot = F.one_hot(real_labels, num_classes=10).float()
            d_out_real = discriminator(real_images, real_onehot)
            d_loss_real = criterion(d_out_real, torch.ones_like(d_out_real) * 0.9)  # label smoothing

            # 2. Fake images with the labels G was given -> should be 0
            fake_labels = torch.randint(0, 10, (batch_size,), device=device)
            fake_onehot = F.one_hot(fake_labels, num_classes=10).float()
            noise = torch.randn(batch_size, noise_dim, device=device)
            g_input = torch.cat([noise, fake_onehot], dim=1)

            fake_images = generator(g_input)
            d_out_fake = discriminator(fake_images.detach(), fake_onehot)
            d_loss_fake = criterion(d_out_fake, torch.zeros_like(d_out_fake))

            d_loss = d_loss_real + d_loss_fake

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train Generator

            d_out_fake_for_g = discriminator(fake_images, fake_onehot)
            g_loss = criterion(d_out_fake_for_g, torch.ones_like(d_out_fake_for_g))

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            bar.unit = f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"

        torch.save(generator.state_dict(), f"{output_dir}/generator_epoch_{epoch + 1}.pth")
