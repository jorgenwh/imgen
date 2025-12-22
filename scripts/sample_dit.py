import os
import torch
import argparse
from tqdm import tqdm
from torchvision.utils import save_image

from imgen.models.dit import DiT
from imgen.training.diffusion import NoiseSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from trained DiT model")

    parser.add_argument(
        "-model",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "-digit",
        type=int,
        required=True,
        choices=range(10),
        help="Digit to generate (0-9)",
    )
    parser.add_argument(
        "-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (must match training)",
    )
    parser.add_argument(
        "-output_path",
        type=str,
        default="",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for sampling",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.isfile(args.model), f"Could not find model file '{args.model}': does not exist."

    device = torch.device(args.device)

    model = DiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        embed_dim=256,
        num_heads=4,
        num_layers=6,
        vocab_size=10,
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    noise_schedule = NoiseSchedule(timesteps=args.timesteps).to(device)

    label = torch.tensor([args.digit], device=device)
    snapshots = []

    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(1, 1, 28, 28, device=device)
        snapshots.append(x.clone())

        # Reverse diffusion: t = T-1, T-2, ..., 0
        for t in tqdm(reversed(range(args.timesteps)), total=args.timesteps, desc="Sampling"):
            t_batch = torch.tensor([t], device=device, dtype=torch.long)

            # Get schedule values
            alpha = noise_schedule.alphas[t_batch].view(1, 1, 1, 1)
            alpha_bar = noise_schedule.alpha_bars[t_batch].view(1, 1, 1, 1)
            beta = noise_schedule.betas[t_batch].view(1, 1, 1, 1)

            # Predict noise
            predicted_noise = model(x, t_batch.float(), label)

            # DDPM reverse step
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise)

            if t > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean

            if t % 100 == 0:
                snapshots.append(x.clone())

    # Stack snapshots: [11, 1, 28, 28]
    snapshots = torch.cat(snapshots, dim=0).cpu()

    # Save as grid (11 images in a row: noise -> final)
    output_path = args.output_path or "outputs"
    os.makedirs(output_path, exist_ok=True)
    img_path = os.path.join(output_path, f"dit_digit_{args.digit}_progression.png")

    save_image(snapshots, img_path, nrow=11, normalize=True, value_range=(-1, 1))
    print(f"Saved diffusion progression to '{img_path}'")


if __name__ == "__main__":
    main()
