from torch import nn
import omegaconf
import torch
from einops import repeat


class TransformerDecoder (nn.Module):
    def __init__(self, configs):
            super().__init__()
            d_model = configs["d_model"]
            nhead = configs["nhead"]
            num_layers = configs["num_layers"]
            self.num_pos_embeddings = configs["num_pos_embeddings"]
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.net = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
            self.pos_embeddings = nn.Embedding(self.num_pos_embeddings, d_model)
            

    def forward(self, x):
          self.bs = x.shape[0]
          positions = torch.arange(self.num_pos_embeddings).to(x.device)
          embeddings = self.pos_embeddings(positions)
          embeddings = repeat(embeddings, 'n d -> repeat n d', repeat=self.bs)
          x = self.net(embeddings, x)
          return x
            
            
            
        