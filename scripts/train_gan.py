import os
import torch
import argparse

from imgen.training.gan_training import train_gan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample GAN Generator")

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
        "-learning_rate",
        type=float,
        default=0.0002,
        help="Learning rate for both the generator and discriminator optimizers",
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "-output_dir",
        type=str,
        default="./models",
        help="Directory to save generator model checkpoints",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_gan(
        noise_dim=args.noise_dim,
        num_channels=args.num_channels,
        width=args.width,
        height=args.height,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
