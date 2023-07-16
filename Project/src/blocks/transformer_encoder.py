from torch import nn
import omegaconf

# from .vision_transformer import (
#     vit_deit_base_distilled_patch16_224,
#     vit_deit_small_distilled_patch16_224,
#     vit_deit_base_distilled_patch16_384,
#     vit_deit_tiny_distilled_patch16_224,
#     vit_large_patch16_224_in21k,
#     vit_base_patch16_224_in21k
# )


class TransformerEncoder (nn.Module):
    def __init__(self, configs):
            super().__init__()
            d_model = configs["d_model"]
            nhead = configs["nhead"]
            num_layers = configs["num_layers"]
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.net = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
            

    def forward(self, x):
          x = self.net(x)
          return x
            
            
            
        
        
   
            
           
        
            
    
