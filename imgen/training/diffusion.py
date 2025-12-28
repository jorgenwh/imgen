import os
import random

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
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


@torch.no_grad()
def sample_dit(
    model: DiT,
    noise_schedule: NoiseSchedule,
    labels: torch.Tensor,
    img_size: int,
    in_channels: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate images from the model using DDPM sampling.

    Args:
        model: The DiT model
        noise_schedule: The noise schedule
        labels: Conditioning labels/tokens, shape (B, ...)
        img_size: Size of images to generate
        in_channels: Number of channels
        device: Device to run on

    Returns:
        Generated images tensor of shape (B, C, H, W)
    """
    model.eval()
    batch_size = labels.shape[0]

    x = torch.randn(batch_size, in_channels, img_size, img_size, device=device)

    for t in tqdm(reversed(range(noise_schedule.timesteps)), total=noise_schedule.timesteps, desc="Sampling"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        alpha = noise_schedule.alphas[t_batch].view(batch_size, 1, 1, 1)
        alpha_bar = noise_schedule.alpha_bars[t_batch].view(batch_size, 1, 1, 1)
        beta = noise_schedule.betas[t_batch].view(batch_size, 1, 1, 1)

        predicted_noise = model(x, t_batch.float(), labels)

        mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise)

        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean

    return x


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
    tokenizer=None,
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

    # Sample 4 random prompts/labels from the dataset for visualization
    # We'll generate 2 images per prompt (8 total) after each epoch
    sample_labels = None
    sample_captions = None
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        print(f"Epoch: {epoch + 1}/{epochs}")
        bar = tqdm(dataloader, desc="Training", unit="batch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} - {unit}")

        for images, labels in bar:
            images = images.to(device) # (B, C, H, W)
            labels = labels.to(device) # (B,) or (B, seq_len) for text
            B, _, _, _ = images.shape

            # On first batch, pick 4 random labels for sample generation
            if sample_labels is None and B >= 4:
                indices = random.sample(range(B), 4)
                # Duplicate each label twice to get 8 samples (2 per prompt)
                sample_labels = torch.cat([labels[indices], labels[indices]], dim=0)
                # Store captions for logging
                if tokenizer is not None:
                    sample_captions = [tokenizer.decode(labels[i].tolist()) for i in indices]
                else:
                    sample_captions = [str(labels[i].item()) for i in indices]

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

            break

        torch.save(model.state_dict(), f"{output_dir}/dit_epoch_{epoch + 1}.pth")

        # Generate sample images for this epoch
        if sample_labels is not None:
            print(f"Generating {sample_labels.shape[0]} sample images...")
            samples = sample_dit(
                model=model,
                noise_schedule=noise_schedule,
                labels=sample_labels,
                img_size=img_size,
                in_channels=in_channels,
                device=device,
            )
            # Save as a grid: 4 columns (one per prompt), 2 rows (2 samples per prompt)
            sample_path = os.path.join(samples_dir, f"epoch_{epoch + 1}.png")

            # Create grid and convert to PIL
            grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = (grid_np * 255).astype("uint8")
            # Handle grayscale (1 channel) images
            if grid_np.shape[2] == 1:
                grid_np = grid_np.squeeze(2)
                grid_img = Image.fromarray(grid_np, mode="L").convert("RGB")
            else:
                grid_img = Image.fromarray(grid_np)

            # Add captions above the image
            if sample_captions is not None:
                col_width = grid_img.width // 4
                try:
                    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 10)
                except OSError:
                    font = ImageFont.load_default()

                # Calculate chars per line based on column width
                chars_per_line = max(10, (col_width - 10) // 6)  # ~6px per char at size 10
                num_lines = 3
                caption_height = 12 * num_lines + 10

                new_img = Image.new("RGB", (grid_img.width, grid_img.height + caption_height), (255, 255, 255))
                new_img.paste(grid_img, (0, caption_height))
                draw = ImageDraw.Draw(new_img)

                for i, caption in enumerate(sample_captions):
                    # Wrap caption to multiple lines
                    lines = []
                    remaining = caption.strip()
                    for _ in range(num_lines):
                        if len(remaining) <= chars_per_line:
                            lines.append(remaining)
                            break
                        lines.append(remaining[:chars_per_line])
                        remaining = remaining[chars_per_line:]
                    else:
                        if remaining:
                            lines[-1] = lines[-1][:-3] + "..."

                    x = i * col_width + 4
                    for j, line in enumerate(lines):
                        draw.text((x, 4 + j * 12), line, fill=(0, 0, 0), font=font)
                new_img.save(sample_path)
            else:
                grid_img.save(sample_path)

            print(f"Saved samples to {sample_path}")
            model.train()  # Set back to training mode
