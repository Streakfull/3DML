from torch import nn
from losses.dice_loss import DiceLoss
from losses.dice_loss_bce import DiceLossBCE
import torch



class BuildLoss():
    def __init__(self, configs):
        self.configs = configs
        self.set_pos_weight()

    def set_pos_weight(self):
        pos_weight = self.configs["pos_weight"]
        if(pos_weight == "None"):
            self.pos_weight = None
        else:
            self.pos_weight = pos_weight


    def get_loss(self):
       match self.configs["criterion"]:
           case "BCE":
               if(self.pos_weight == None):
                   return nn.BCEWithLogitsLoss()
               else:
                   return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
           case "DICE":
               return DiceLoss()
       return DiceLossBCE(pos_weight=self.pos_weight)
               
           
