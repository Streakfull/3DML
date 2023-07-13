from torch import nn
import omegaconf
import torch
from einops import repeat


class TransformerDecoder (nn.Module):
    def __init__(self, configs_path="./configs/transformer_configs.yaml" ):
            super().__init__()
            self.sequential = nn.Sequential()
            decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=12, batch_first=True)
            self.net = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=8)
            self.pos_embeddings = nn.Embedding(64 ,768)
            

    def forward(self, x):
          self.bs = x.shape[0]
          positions = torch.arange(64).to(x.device)
          embeddings = self.pos_embeddings(positions)
          embeddings = repeat(embeddings, 'n d -> repeat n d', repeat=self.bs)
          x = self.net(embeddings,x)
          return x
            
            
            
        