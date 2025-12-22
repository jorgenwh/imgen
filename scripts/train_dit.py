import os
import torch
import argparse

from imgen.training.diffusion import train_dit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DiT diffusion model")

    parser.add_argument(
        "-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "-learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=10,
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
        help="Directory to save model checkpoints",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_dit(
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
