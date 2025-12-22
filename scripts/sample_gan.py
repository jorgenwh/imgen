import os
import torch
import argparse
from torchvision.utils import save_image

from imgen.models.gan import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample GAN Generator")

    parser.add_argument(
        "-model",
        type=str,
        required=True,
        help="Path to the pre-trained generator model",
    )
    parser.add_argument(
        "-noise_dim",
        type=int,
        default=128,
        help="Dimensionality of the noise vector input to the generator",
    )
    parser.add_argument(
        "-num_channels",
        type=int,
        default=1,
        help="Number of channels in the generated images (1 for grayscale, 3 for RGB)",
    )
    parser.add_argument(
        "-width",
        type=int,
        default=28,
        help="Width of the generated images",
    )
    parser.add_argument(
        "-height",
        type=int,
        default=28,
        help="Height of the generated images",
    )
    parser.add_argument(
        "-num_samples",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "-output_path",
        type=str,
        default="",
        help="Path to save the generated images",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.isfile(args.model), f"Could not find model file '{args.model}': does not exist."

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    device = torch.device(args.device)

    generator = Generator(
        noise_dim=args.noise_dim,
        num_channels=args.num_channels,
        width=args.width,
        height=args.height,
    ).to(device)
    generator.load_state_dict(torch.load(args.model, map_location=device))
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(args.num_samples, args.noise_dim, device=device)
        fake_images = generator(noise)

        # Save as grid
        output_path = args.output_path or "outputs"
        os.makedirs(output_path, exist_ok=True)
        img_path = os.path.join(output_path, "samples.png")

        # save_image handles [-1,1] to [0,1] normalization
        save_image(fake_images, img_path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"Saved {args.num_samples} samples to '{img_path}'")


if __name__ == "__main__":
    main()
