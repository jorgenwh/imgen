import os
import torch
import argparse

from imgen.data.datasets import get_mnist_dataloader, get_coco_dataloader, CharTokenizer
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
    parser.add_argument(
        "-experiment",
        type=str,
        default="mnist",
        help="Which experiment to run (mnist, coco)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiment == "mnist":
        print("Starting DiT training on MNIST")
        img_size = 28
        patch_size = 4
        in_channels = 1
        embed_dim = 256
        num_heads = 4
        num_layers = 6
        vocab_size = 10
        dataloader = get_mnist_dataloader(batch_size=args.batch_size, data_dir="./data")
    elif args.experiment == "coco":
        print("Starting DiT training on COCO")
        img_size = 128
        patch_size = 8
        in_channels = 3
        embed_dim = 1024
        num_heads = 16
        num_layers = 24
        tokenizer = CharTokenizer(max_len=128)
        vocab_size = tokenizer.vocab_size
        dataloader = get_coco_dataloader(
            batch_size=args.batch_size,
            img_size=img_size,
            tokenizer=tokenizer,
            data_dir="./data/coco",
        )
    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")

    train_dit(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        dataloader=dataloader,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=torch.device(args.device),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
