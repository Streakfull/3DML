from torch import nn
import omegaconf
import torch
from einops import repeat, rearrange
import numpy as np


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
            self.layer_norm = nn.LayerNorm(d_model)
            #self.rng = np.random.default_rng()

    def forward(self, x):
          self.bs = x.shape[0]
          #positions = torch.arange(self.num_pos_embeddings).to(x.device)
          self.get_gen_order()
          embeddings = self.pos_embeddings(self.gen_order.to(x.device))
          #embeddings = rearrange(embeddings, 'bs sq d -> (bs sq) d')
          #y = self.gen_order.flatten()
          #sorted_output = torch.zeros_like(embeddings)
          #sorted_output[y] = embeddings
          #sorted_output = rearrange(embeddings, '(bs sq) d -> bs sq d', bs=self.bs, sq=self.num_pos_embeddings)
          #embeddings = repeat(embeddings, 'n d -> repeat n d', repeat=self.bs)
          #import pdb;pdb.set_trace()  
          embeddings = repeat(embeddings, 'n d -> repeat n d', repeat=self.bs)   
          x = self.net(embeddings, x)
          x = self.layer_norm(x)
            
          return x
    
    def get_gen_order(self):
        self.gen_order = np.arange(self.num_pos_embeddings)
        repeated = repeat(self.gen_order, 'd -> repeat d', repeat=self.bs)
        #self.gen_order = self.rng.permuted(repeated, axis=1)
        self.gen_order = torch.tensor(self.gen_order)
       
        
        
        
        
            
            
        