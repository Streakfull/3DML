import math
import torch.nn.functional as F
import torch.nn as nn
#from empatches import EMPatches
import torch
from einops import rearrange,repeat
import transformers
from cprint import *

# from timm.models.vision_transformer import (
#         HybridEmbed,
#         PatchEmbed)

from timm.models.vision_transformer import PatchEmbed
from timm.models.vision_transformer_hybrid import HybridEmbed   

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
      #self.patch_embedding = nn.Linear(in_features=self.in_features,out_features=self.embedding_dim)
      self.patch_embed = PatchEmbed(
                img_size=137, patch_size=13, in_chans=4, embed_dim=768)

    def forward(self, x):
      x = self.patch_embed(x)
    
      #self.set_input(images)
#       embedded_patches = self.patch_embedding(self.patches)
      positions = torch.arange(self.N).to(x.device)
      pos_embedding = self.pos_embedding(positions)
      x = x + pos_embedding   
       #embedding = embedded_patches[:,self.random_indices,:] + pos_embedding[self.random_indices]
      #import pdb;pdb.set_trace()
      return x
       


    def patch(self, x, kernel_size=13, stride=13):
      pd = (self.padding,self.padding,self.padding,self.padding)
      x = F.pad(x, pd)
      x_cubes = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
      x_cubes = rearrange(x_cubes, 'b c p1 p2 h w -> b c (p1 p2 ) h w')
      x_cubes = rearrange(x_cubes, 'b c p  h w -> b p c h w')
      return x_cubes
    
    def set_input(self, images):
#        self.patches = self.patch(images,kernel_size=self.patch_size, stride=self.patch_size).float() # bs x 4 x 13 x 13
#        self.patches = self.patches.flatten(start_dim=2)
       self.random_indices = torch.randperm(self.N)
     
       