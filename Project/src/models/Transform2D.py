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


class Transform2D(BaseModel):
    def __init__(self, configs_path="./configs/global_configs.yaml"):
        super().__init__()
        configs = omegaconf.OmegaConf.load(configs_path)["model"]
        self.patch_encoder = PatchEncoder()
        self.transformer_encoder = TransformerEncoder()
        self.transformer_decoder = TransformerDecoder()
        #self.decoder = Decoder(in_channels=768)
        self.decoder = SimpleDecoder()
        
        #self.encoder = Encoder(in_channels=4)
        # self.transformer = Transformer()
        # self.decoder = Decoder(in_channels=256)
        self.criterion_demo = torch.nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor(1.8))
        # #self.cireterion = nn.
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
      #target_repeated = torch.repeat_interleave(target,bs,dim=0)
      #import pdb;pdb.set_trace()
      self.loss = self.criterion(self.x, target)/bs
      self.loss_demo = self.criterion_demo(self.sigmoid(self.x),target)
      self.loss.backward()
      #self.loss_demo.backward()
      # Move target to device here


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
    #def to(device):
        


     