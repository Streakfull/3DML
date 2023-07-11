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
            self.linear1tf = nn.Linear(1024,1024*2)
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=1024*2, nhead=nhead, batch_first=True)
            self.net2 = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
            self.linear2 = nn.Linear(1024*2,1024*2*2)
            self.linear2tf = nn.Linear(1024*2,1024*2*2)
            
            #encoder_layer = nn.TransformerEncoderLayer(d_model=1024*2*2, nhead=nhead, batch_first=True)
            #self.net3 = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
            self.linear3 = nn.Linear(1024*2*2,1024*2*2*2)
            self.linear3tf = nn.Linear(1024*2*2,1024*2*2*2)
            
            
            #encoder_layer = nn.TransformerEncoderLayer(d_model=1024*2*2*2, nhead=nhead, batch_first=True)
            #self.net4 = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
            self.linear4 = nn.Sequential(nn.Linear(1024*2*2*2,12288),nn.LeakyReLU(0.1),nn.Linear(12288,16384))
            
            
            
            d_model = d_model * 2
            #self.sequential.append(net)
            
#             for l in range(1,num_transformers):
#               encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#               net = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
#               d_model_next = d_model * 2
#               linear1 = nn.Linear(d_model,d_model_next)
#               activation = nn.LeakyReLU(0.1)
#               linear2 = nn.Linear(d_model_next,d_model_next)
#               d_model = d_model_next
#               self.sequential.append(net)
#               self.sequential.append(linear1)
#               #self.sequential.append(activation)
#               #self.sequential.append(linear2)

#             for i in range(num_linear_layers):
#                  d_model_next = d_model * 2
#                  linear = nn.Linear(d_model,d_model_next)
#                  activation = nn.Softmax(dim=-1)
#                  d_model = d_model_next
#                  self.sequential.append(linear)
#                  if(i != num_linear_layers-1):
#                     self.sequential.append(activation)
         

    def forward(self, x):
          input_l1 = self.linear1(x) #2048
          x = self.net(x) #1024
          x = self.linear1tf(x) + input_l1 #2048
          tf1 = x #2048
          x = self.net2(x) # 2048
          x +=tf1
           
          input_l2 = self.linear2(input_l1) #4096 
          x = self.linear2tf(x) + input_l2 #4096
          #tf2 = x
          #x = self.net3(x)
          #x+=tf2
        
          input_l3 = self.linear3(input_l2) #8192
          x = self.linear3tf(x) + input_l3 # 8192
          #tf3 = x
          #x = self.net4(x)
          #x+=tf3
          x = self.linear4(x)
          #import pdb;pdb.set_trace()
          return x
            
            
            
            
        
        
   
            
           
        
            
    
