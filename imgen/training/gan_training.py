import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from imgen.data.datasets import get_mnist_dataloader
from imgen.models.gan import Discriminator, Generator, weights_init


def train_gan(
        noise_dim: int,
        num_channels: int,
        width: int,
        height: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        device: torch.device,
        output_dir: str
):
    dataloader = get_mnist_dataloader(batch_size=batch_size, data_dir="./data")

    discriminator = Discriminator(num_channels, width, height).to(device)
    generator = Generator(noise_dim, num_channels, width, height).to(device)

    # Apply DCGAN weight initialization
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1) + "/" + str(epochs))
        bar = tqdm(dataloader, desc="Training", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} - {unit}")
        for real_images, _ in bar:
            real_images = real_images.to(device)

            # Train discriminator
            d_out_real = discriminator(real_images)
            real_labels = torch.ones_like(d_out_real)

            noise = torch.randn(real_images.size(0), noise_dim).to(device)
            fake_images = generator(noise)
            d_out_fake = discriminator(fake_images.detach()) # Detach to avoid training G on these labels
            fake_labels = torch.zeros_like(d_out_fake)

            d_loss_real = criterion(d_out_real, real_labels)
            d_loss_fake = criterion(d_out_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            d_out_fake_for_g = discriminator(fake_images)
            g_loss = criterion(d_out_fake_for_g, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            bar.unit = f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"

        torch.save(generator.state_dict(), f"{output_dir}/generator_epoch_{epoch + 1}.pth")
