import math
import torch.nn.functional as F
import torch.nn as nn
#from empatches import EMPatches
import torch
from einops import rearrange,repeat

class PatchEncoder(nn.Module):
    def __init__(self):
      super(PatchEncoder, self).__init__()
      self.in_features = 13 * 13 * 4
      self.N = 121
      self.embedding_dim = 768
      self.pos_embedding = nn.Embedding(self.N, self.embedding_dim)
      self.patch_embedding = nn.Linear(in_features=self.in_features,out_features=self.embedding_dim)

    def forward(self, images):
      self.set_input(images)
      embedded_patches = self.patch_embedding(self.patches)
      positions = torch.arange(self.N).to(images.device)
      pos_embedding = self.pos_embedding(positions)
      embedding = embedded_patches + pos_embedding
      return embedding
       


    def patch(self, x, kernel_size=13, stride=13):
      pd = (3,3,3,3)
      x = F.pad(x,pd)
      x_cubes = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
      x_cubes = rearrange(x_cubes, 'b c p1 p2   h w -> b c (p1 p2 )  h w')
      x_cubes = rearrange(x_cubes, 'b c p  h w -> b p c h w')
      return x_cubes
    
    def set_input(self, images):
       self.patches = self.patch(images).float() # bs x 4 x 13 x 13
       self.patches = self.patches.flatten(start_dim=2)
     
       