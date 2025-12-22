import torch
import PIL

from imgen.models.gan import Generator

def generate_images(num_images: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_dim=100, num_channels=1, width=28, height=28).to(device)
    generator.load_state_dict(
        torch.load("generator_epoch_10.pth", map_location=device)
    )

    noise = torch.randn(num_images, 100).to(device)
    generated_images = generator(noise)

    # Post-process and save images
    for i in range(num_images):
        img = generated_images[i].cpu().detach()
        img = (img + 1) / 2  # Rescale to [0, 1]
        img = img.clamp(0, 1)
        img = img.squeeze(0)  # Remove channel dimension for grayscale
        img = PIL.Image.fromarray((img.numpy() * 255).astype('uint8'), mode='L')
        img.save(f"outputs/generated_image_{i + 1}.png")
