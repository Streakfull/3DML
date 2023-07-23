from collections import OrderedDict
from models.base_model import BaseModel
# from blocks.torch_decoder import Decoder
# from blocks.patch_encoder import PatchEncoder
import torch
from torch import nn,optim
from einops import rearrange
import omegaconf
from losses.build_loss import BuildLoss
from blocks.transformer_decoder import TransformerDecoder
from blocks.simple_decoder import SimpleDecoder
from utils.util import iou
from transformers import  DeiTModel



# TODO: Handle criterion from configs
class Transform2D(BaseModel):
    def __init__(self, configs_path="./configs/global_configs.yaml"):
        super().__init__()
        configs = omegaconf.OmegaConf.load(configs_path)["model"]
        #self.patch_encoder = PatchEncoder(configs["encoder"])
        #self.transformer_encoder = TransformerEncoder(configs["transformer_encoder"])
        self.transformer_decoder = TransformerDecoder(configs["transformer_decoder"])
        self.decoder = SimpleDecoder()
        self.criterion = BuildLoss(configs).get_loss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.5)
        self.sigmoid = torch.nn.Sigmoid()
        self.deit_model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.deit_model.eval()

    def set_input(self, input):
        self.images = input['images']
        self.voxels = input['voxels']
        bs, nimgs, h, w, c = self.images.shape
        self.bs = bs
        self.images = rearrange(self.images,'bs nimgs c h w -> (bs nimgs) c h w')
        self.nimgs = nimgs
        #self.images = torch.Tensor(self.images)
      
    

    def forward(self, x):
        self.set_input(x)
        ## Encode
        x = self.deit_model(self.images).last_hidden_state
        ## Fusion
        if(self.nimgs > 1):
             x = rearrange(x, '(bs nimgs) s p -> bs nimgs s p', bs=self.bs,nimgs=self.nimgs)
             x = x.mean(dim=1)
        x = self.transformer_decoder(x)
        ## Occupancy Grid
        x = rearrange(x,'bs (c1 c2 c3) d -> bs d c1 c2 c3', c1=4,c2=4,c3=4)
        x = self.decoder(x)
        self.x = x.squeeze(1)
        return x
    
    def backward(self):
      target = self.voxels.squeeze(1)
      self.loss = self.criterion(self.x, target)
     
    
    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.loss.backward()
        self.optimizer.step()
        
    
    def get_metrics(self):
        iou_val = self.get_iou()
        return  OrderedDict([
            ('loss', self.loss.data),
            ('iou', iou_val),
           
        ])
    
    def get_iou(self):
        gt =  target = self.voxels.squeeze(1)
        iou_val = iou(gt, self.x, 0.5)
        return iou_val.mean()
    
    def inference(self, x):
        self.eval()
        x = self.forward(x)
        self.backward()
        
        


     