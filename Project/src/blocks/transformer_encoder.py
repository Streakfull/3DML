from torch import nn
import omegaconf


class TransformerEncoder (nn.Module):
    def __init__(self, configs_path="./configs/transformer_configs.yaml" ):
            super().__init__()
            configs = omegaconf.OmegaConf.load(configs_path)
            d_model = configs["d_model"]
            nhead = configs["nhead"]
            num_layers = configs["num_layers"]
            num_transformers = configs['num_transformers']
            num_linear_layers = configs["num_linear_layers"]
            self.sequential = nn.Sequential()
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.net = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
            

    def forward(self, x):
          x = self.net(x)
          return x
            
            
            
        
        
   
            
           
        
            
    
