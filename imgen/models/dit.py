import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 28, patch_size: int = 4, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: [B, C, H, W] -> [B, num_patches, patch_size^2 * C]
        x = x.unfold(2, p, p).unfold(3, p, p)  # [B, C, H//p, W//p, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5)  # [B, H//p, W//p, C, p, p]
        x = x.reshape(B, self.num_patches, -1)  # [B, num_patches, C*p*p]

        return self.proj(x)  # [B, num_patches, embed_dim]


class Unpatchify(nn.Module):
    def __init__(self, img_size: int = 28, patch_size: int = 4, out_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.out_channels = out_channels

        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_patches, embed_dim]
        B = x.shape[0]
        p = self.patch_size
        g = self.grid_size

        x = self.proj(x)  # [B, num_patches, p*p*C]
        x = x.reshape(B, g, g, self.out_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, g, p, g, p]
        x = x.reshape(B, self.out_channels, self.img_size, self.img_size)

        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] (patches)
        # context: [B, M, D] (conditioning, e.g., text)
        B, N, D = x.shape
        M = context.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = CrossAttention(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x, x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = SelfAttention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        # Cross-attention to conditioning
        x = x + self.cross_attn(self.norm2(x), context)
        # Feedforward
        x = x + self.ff(self.norm3(x))
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for MNIST.

    Takes a noisy image, timestep, and text label (digit 0-9),
    predicts the noise.
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        vocab_size: int = 10,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Text embedding (digit 0-9 -> embedding)
        self.text_embed = nn.Embedding(vocab_size, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.unpatchify = Unpatchify(img_size, patch_size, in_channels, embed_dim)

        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy images [B, C, H, W]
            t: Timesteps [B]
            label: Class labels [B] (digits 0-9)

        Returns:
            Predicted noise [B, C, H, W]
        """
        # Embed patches
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        x = x + self.pos_embed

        # Embed timestep and add to patches
        t_emb = self.time_embed(t)  # [B, embed_dim]
        x = x + t_emb.unsqueeze(1)  # broadcast to all patches

        # Embed text label as conditioning
        context = self.text_embed(label).unsqueeze(1)  # [B, 1, embed_dim]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, context)

        x = self.norm(x)

        # Reconstruct image
        return self.unpatchify(x)
