import os
import torch
import torch.nn.functional as F
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

    generator = Generator(noise_dim=args.noise_dim + 10).to(device)
    generator.load_state_dict(torch.load(args.model, map_location=device))
    generator.eval()

    with torch.no_grad():
        digits = torch.arange(10)
        labels = F.one_hot(digits, num_classes=10).float()
        noise = torch.randn(10, args.noise_dim)
        conditional_input = torch.cat((noise, labels), dim=1)
        conditional_input = conditional_input.to(device)

        fake_images = generator(conditional_input).cpu()

        # Save as grid
        output_path = args.output_path or "outputs"
        os.makedirs(output_path, exist_ok=True)
        img_path = os.path.join(output_path, "samples.png")

        # save_image handles [-1,1] to [0,1] normalization (2x5 grid)
        save_image(fake_images, img_path, nrow=5, normalize=True, value_range=(-1, 1))
        print(f"Saved {len(fake_images)} samples to '{img_path}'")


if __name__ == "__main__":
    main()
