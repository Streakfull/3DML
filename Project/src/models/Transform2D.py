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
from utils.util import iou



class DiceLoss(nn.Module):
    """based on https://github.com/hubutui/DiceLoss-PyTorch"""

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = torch.sigmoid(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class CEDiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean'):
        super(CEDiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, output, target):
        ce_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return ce_loss + dice_loss
# TODO: Handle criterion from configs
class Transform2D(BaseModel):
    def __init__(self, configs_path="./configs/global_configs.yaml"):
        super().__init__()
        configs = omegaconf.OmegaConf.load(configs_path)["model"]
        self.patch_encoder = PatchEncoder(configs["encoder"])
        #self.patch_encoder = Encoder()
        self.transformer_encoder = TransformerEncoder(configs["transformer_encoder"])
        self.transformer_decoder = TransformerDecoder(configs["transformer_decoder"])
        self.decoder = SimpleDecoder()
        #self.criterion_demo = torch.nn.BCELoss()
        #self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(configs["pos_weight"]))
        #self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.8))
        self.criterion = CEDiceLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def set_input(self, input):
        self.images = input['images']
        self.voxels = input['voxels']
        bs, nimgs, h, w, c = self.images.shape
        self.bs = bs
        self.images = rearrange(self.images,'bs nimgs c h w -> (bs nimgs) c h w')
        self.nimgs = nimgs
        #self.images = torch.Tensor(self.images)
      
    

    def forward(self, x):
        ## Encode
        self.set_input(x)
        x = self.patch_encoder(self.images)  # bs x n_patches x 768
        if(self.nimgs > 1):
            x = rearrange(x, '(bs nimgs) s p -> bs nimgs s p', bs=self.bs,nimgs=self.nimgs)
            x = x.mean(dim=1)
        x = self.transformer_encoder(x)
        x = self.transformer_decoder(x)
        x = rearrange(x,'bs (c1 c2 c3) d -> bs d c1 c2 c3', c1=4,c2=4,c3=4)
        x = self.decoder(x)
        self.x = x.squeeze(1)
        return x
    
    def backward(self):
      bs, c1, c2, c3 = self.x.shape
      target = self.voxels.squeeze(1)
      #self.loss = self.criterion(self.x, target)
      #self.loss = self.dice_loss(self.x, target)
      self.loss = self.criterion(self.x, target)
      #import pdb;pdb.set_trace()
      #self.loss_demo = self.criterion_demo(self.sigmoid(self.x),target)
     
    
    
    def dice_loss(self,logits,labels):
#         self.bs = inp.shape[0]
#         intersection = (inp * target).sum() 
#         negative_intersection = ((1-inp)*(1-target)).sum()
#         summation = (inp+target).sum()
#         summation_right = ((2-inp)-target).sum()
#         loss =  1 - (intersection/summation) - (negative_intersection/summation_right)
#         return loss
        self.p = 1
        self.smooth = 1
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss/self.bs

        

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
        #import pdb;pdb.set_trace()
        gt =  target = self.voxels.squeeze(1)
        iou_val = iou(gt, self.x, 0.5)
        return iou_val.mean()
    
    def inference(self, x):
        self.eval()
        x = self.forward(x)
        self.backward()
        
        


     