import torch
import torch.nn.functional as F
from tqdm import tqdm

from imgen.models.dit import DiT


class NoiseSchedule:
    """
    Linear noise schedule for diffusion.
    Uses the "alpha" and "alpha_bar" parameterization from DDPM paper.
    """
    def __init__(self, timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


def train_dit(
    img_size: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    timesteps: int,
    learning_rate: float,
    epochs: int,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    output_dir: str,
):
    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    noise_schedule = NoiseSchedule(timesteps=timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        print(f"Epoch: {epoch + 1}/{epochs}")
        bar = tqdm(dataloader, desc="Training", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} - {unit}")

        for images, labels in bar:
            images = images.to(device) # (B, C, H, W)
            labels = labels.to(device) # (B,)
            B, _, _, _ = images.shape

            # Sample random timesteps (B,)
            t = torch.randint(0, noise_schedule.timesteps, (B,), device=device)

            # Add noise: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
            noise = torch.randn_like(images)
            alpha_bar = noise_schedule.alpha_bars[t].view(B, 1, 1, 1)
            noisy_images = torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * noise

            # Predict the noise
            predicted_noise = model(noisy_images, t.float(), labels)

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            bar.unit = f"Loss: {total_loss / num_batches:.4f}"

        torch.save(model.state_dict(), f"{output_dir}/dit_epoch_{epoch + 1}.pth")
