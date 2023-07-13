from collections import OrderedDict
from models.base_model import BaseModel
from blocks.torch_encoder import Encoder
from blocks.transformer_encoder import TransformerEncoder
from blocks.torch_decoder import Decoder
import torch
from torch import nn,optim
from einops import rearrange
import omegaconf
from blocks.patch_encoder import PatchEncoder
from blocks.transformer_decoder import TransformerDecoder
from blocks.simple_decoder import SimpleDecoder

# TODO: Handle criterion from configs
class Transform2D(BaseModel):
    def __init__(self, configs_path="./configs/global_configs.yaml"):
        super().__init__()
        configs = omegaconf.OmegaConf.load(configs_path)["model"]
        self.patch_encoder = PatchEncoder(configs["encoder"])
        self.transformer_encoder = TransformerEncoder(configs["transformer_encoder"])
        self.transformer_decoder = TransformerDecoder(configs["transformer_decoder"])
        self.decoder = SimpleDecoder()
        self.criterion_demo = torch.nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(configs["pos_weight"]))
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def set_input(self, input):
        self.images = input['images']
        self.voxels = input['voxels']
        bs, nimgs, h, w, c = self.images.shape
        self.bs = bs
        self.images = rearrange(self.images,'bs nimgs c h w -> (bs nimgs) c h w')
        #self.images = torch.Tensor(self.images)
      
    

    def forward(self, x):
        ## Encode
        self.set_input(x)
        x = self.patch_encoder(self.images)  # bs x n_patches x 768
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(x)
        x = rearrange(x,'bs (c1 c2 c3) d -> bs d c1 c2 c3', c1=4,c2=4,c3=4)
        x = self.decoder(x)
        self.x = x.squeeze(1)
        return x
    
    def backward(self):
      bs, c1, c2, c3 = self.x.shape
      target = self.voxels.squeeze(1)
      self.loss = self.criterion(self.x, target)
      self.loss_demo = self.criterion_demo(self.sigmoid(self.x),target)
      self.loss.backward()


    def step(self, x):
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()
        
    
    def get_loss(self):
        return  OrderedDict([
            ('loss', self.loss.data),
            ('loss_demo',self.loss_demo.data),
           
        ])
        


     