from collections import OrderedDict
from models.base_model import BaseModel
from blocks.torch_encoder import Encoder
from blocks.transformer import Transformer
from blocks.torch_decoder import Decoder
import torch
from torch import nn,optim
from einops import rearrange
import omegaconf


class Transform2D(BaseModel):
    def __init__(self, configs_path="./configs/global_configs.yaml"):
        super().__init__()
        configs = omegaconf.OmegaConf.load(configs_path)["model"]
        self.encoder = Encoder(in_channels=4)
        self.transformer = Transformer()
        self.decoder = Decoder()
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad == True], lr=configs["lr"])

    def set_input(self, input):
        self.images = input['images']
        self.voxels = input['voxels']
        bs, nimgs, h, w, c = self.images.shape
        self.bs = bs
        self.images = rearrange(self.images,'bs nimgs h w c -> (bs nimgs) c h w')
        self.images = torch.Tensor(self.images)
      
    

    def forward(self, x):
        ## Encode
        self.set_input(x)
        x = self.encoder(self.images)  # bs x nimgs x 1024
        # Pass through transformer
        x = self.transformer(x)    # bs x nimgs x 32768
        x = rearrange(x, 'bs (c i j k) -> bs c i j k', c = 256, i=4 , j=4, k=4)
        x = self.decoder(x)

        #import pdb;pdb.set_trace()
        self.x = x
        return x
    
    def backward(self):
      bs, c1, c2, c3 = self.x.shape
      target = self.voxels
      target_repeated = torch.repeat_interleave(target,bs,dim=0)
      self.loss = self.criterion(self.x,target_repeated)
      self.loss.backward()
      # Move target to device here


    def step(self, x):
        x = self.forward(x)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        
    
    def get_loss(self):
        return  OrderedDict([
            ('loss', self.loss.data),
        ])
    #def to(device):
        


     