from torch import nn
import omegaconf


class Transformer (nn.Module):
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
            self.linear1 = nn.Linear(1024,1024*2)
            d_model = d_model * 2
            #self.sequential.append(net)
            
            for l in range(1,num_transformers):
              encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
              net = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
              d_model_next = d_model * 2
              linear1 = nn.Linear(d_model,d_model_next)
              activation = nn.LeakyReLU(0.1)
              linear2 = nn.Linear(d_model_next,d_model_next)
              d_model = d_model_next
              self.sequential.append(net)
              self.sequential.append(linear1)
              #self.sequential.append(activation)
              #self.sequential.append(linear2)

            for i in range(num_linear_layers):
                 d_model_next = d_model * 2
                 linear = nn.Linear(d_model,d_model_next)
                 activation = nn.Softmax(dim=-1)
                 d_model = d_model_next
                 self.sequential.append(linear)
                 if(i != num_linear_layers-1):
                    self.sequential.append(activation)
         

    def forward(self, x):
           shortcut = x
           x = self.net(x)
           x += shortcut
           x  = self.linear1(x)
           x = self.sequential(x)
           #import pdb;pdb.set_trace()
           return x
