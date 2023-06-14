import torch.nn as nn
import torch
from torch.nn.utils import weight_norm as wn


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        self.latent_size = latent_size
        dropout_prob = 0.2
        ln1 = wn(nn.Linear(in_features=latent_size+3,out_features=512))
        ln2 = wn(nn.Linear(in_features=512,out_features=512))
        ln3 = wn(nn.Linear(in_features=512,out_features=512))
        ln4 = wn(nn.Linear(in_features=512,out_features=512-latent_size-3))
        


        self.first_half = nn.Sequential(ln1,nn.ReLU(),nn.Dropout(dropout_prob),
                                   ln2,nn.ReLU(),nn.Dropout(dropout_prob),
                                   ln3,nn.ReLU(),nn.Dropout(dropout_prob),
                                   ln4,nn.ReLU(),nn.Dropout(dropout_prob)
                                   )
        
        ln5 = wn(nn.Linear(in_features=512,out_features=512))
        ln6 = wn(nn.Linear(in_features=512,out_features=512))
        ln7 = wn(nn.Linear(in_features=512,out_features=512))
        ln8 = wn(nn.Linear(in_features=512,out_features=512))
        ln9 = wn(nn.Linear(in_features=512,out_features=1))
        self.second_half =  nn.Sequential(ln5,nn.ReLU(),nn.Dropout(dropout_prob),
                                   ln6,nn.ReLU(),nn.Dropout(dropout_prob),
                                   ln7,nn.ReLU(),nn.Dropout(dropout_prob),
                                   ln8,nn.ReLU(),nn.Dropout(dropout_prob),
                                   ln9
                                   )
        # TODO: Define model

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        #batch_size = x_in.shape(0)
        x_first_half = self.first_half(x_in)
        concat = torch.cat((x_first_half,x_in),dim=1)
        output = self.second_half(concat)
        return output
