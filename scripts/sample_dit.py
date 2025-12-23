import os
import torch
import argparse
from tqdm import tqdm
from torchvision.utils import save_image

from imgen.models.dit import DiT
from imgen.training.diffusion import NoiseSchedule
from imgen.data.datasets import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from trained DiT model")

    parser.add_argument(
        "-model",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "-prompt",
        type=str,
        required=True,
        help="Prompt for generation (digit 0-9 for MNIST, text for COCO)",
    )
    parser.add_argument(
        "-experiment",
        type=str,
        default="mnist",
        help="Which experiment (mnist, coco)",
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

    # Configure based on experiment
    if args.experiment == "mnist":
        img_size, patch_size, in_channels = 28, 4, 1
        embed_dim, num_heads, num_layers = 256, 4, 6
        vocab_size = 10
        tokens = torch.tensor([int(args.prompt)], device=device)
        output_name = f"dit_digit_{args.prompt}_progression.png"
    elif args.experiment == "coco":
        img_size, patch_size, in_channels = 64, 8, 3
        embed_dim, num_heads, num_layers = 512, 8, 12
        tokenizer = CharTokenizer(max_len=128)
        vocab_size = tokenizer.vocab_size
        tokens = torch.tensor([tokenizer.encode(args.prompt)], device=device)  # [1, 128]
        # Sanitize prompt for filename
        safe_prompt = "".join(c if c.isalnum() else "_" for c in args.prompt[:20])
        output_name = f"dit_text_{safe_prompt}_progression.png"
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    noise_schedule = NoiseSchedule(timesteps=args.timesteps).to(device)

    snapshots = []

    with torch.no_grad():
        x = torch.randn(1, in_channels, img_size, img_size, device=device)
        snapshots.append(x.clone())

        for t in tqdm(reversed(range(args.timesteps)), total=args.timesteps, desc="Sampling"):
            t_batch = torch.tensor([t], device=device, dtype=torch.long)

            alpha = noise_schedule.alphas[t_batch].view(1, 1, 1, 1)
            alpha_bar = noise_schedule.alpha_bars[t_batch].view(1, 1, 1, 1)
            beta = noise_schedule.betas[t_batch].view(1, 1, 1, 1)

            predicted_noise = model(x, t_batch.float(), tokens)

            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise)

            if t > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean

            if t % 100 == 0:
                snapshots.append(x.clone())

    snapshots = torch.cat(snapshots, dim=0).cpu()

    output_path = args.output_path or "outputs"
    os.makedirs(output_path, exist_ok=True)
    img_path = os.path.join(output_path, output_name)

    save_image(snapshots, img_path, nrow=11, normalize=True, value_range=(-1, 1))
    print(f"Prompt: '{args.prompt}'")
    print(f"Saved diffusion progression to '{img_path}'")


if __name__ == "__main__":
    main()
