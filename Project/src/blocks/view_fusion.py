from torch import nn
import omegaconf
import torch
from einops import repeat, rearrange
import numpy as np


class ViewFusion (nn.Module):
    def __init__(self, configs):
            super().__init__()
            d_model = configs["d_model"]
            nhead = configs["nhead"]
            num_layers = configs["num_layers"]
            self.num_pos_embeddings = 196
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.net = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
            self.pos_embeddings = nn.Embedding(self.num_pos_embeddings, d_model)
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
          self.bs, self.nimgs, self.seq_length ,  _ = x.shape
          self.get_gen_order()
          embeddings = self.pos_embeddings(self.gen_order.to(x.device))
          x = rearrange(x, 'bs nimgs seq_length d -> bs (nimgs seq_length) d')
          x = self.net(embeddings, x)
          #import pdb; pdb.set_trace()
          x = self.layer_norm(x)
            
          return x
    
    def get_gen_order(self):
        self.gen_order = np.arange(self.num_pos_embeddings)
        self.gen_order = repeat(self.gen_order, 'd -> repeat d', repeat=self.bs)
        #self.gen_order = self.rng.permuted(repeated, axis=1)
        self.gen_order = torch.tensor(self.gen_order)
       
        
        
        
        
            
            
        