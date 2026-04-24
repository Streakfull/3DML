import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed


class PatchEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.channels = 4
        self.patch_size = configs["patch_size"]
        self.N = configs["sequence_length"]
        self.embedding_dim = configs["embedding_dim"]
        self.padding = configs["patch_padding"]
        self.in_features = self.patch_size * self.patch_size * self.channels
        self.pos_embedding = nn.Embedding(self.N, self.embedding_dim)
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=16, in_chans=3, embed_dim=self.embedding_dim)

    def forward(self, x):
        x = self.patch_embed(x)

        self.bs = x.shape[0]
        positions = torch.arange(self.N).to(x.device)
        pos_embedding = self.pos_embedding(positions)
        x = x + pos_embedding
        return x

    def patch(self, x, kernel_size=13, stride=13):
        pd = (self.padding, self.padding, self.padding, self.padding)
        x = F.pad(x, pd)
        x_cubes = x.unfold(2, kernel_size, stride).unfold(
            3, kernel_size, stride)
        x_cubes = rearrange(x_cubes, 'b c p1 p2 h w -> b c (p1 p2 ) h w')
        x_cubes = rearrange(x_cubes, 'b c p  h w -> b p c h w')
        return x_cubes

    def set_input(self, images):
        self.random_indices = torch.randperm(self.N)
