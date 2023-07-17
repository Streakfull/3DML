import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

# TODO: Refactor this properly
class SimpleDecoder(nn.Module):
     def __init__(self):
              super().__init__()
              self.initial_conv = nn.Conv3d(in_channels=768,out_channels=64, kernel_size=3,padding=1)
              self.encoded_conv = nn.Sequential(ResBlock(64),ResBlock(64))
              
              self.conv_transpose = nn.Sequential()
              for i in range(3):
                    self.conv_transpose.append(nn.ConvTranspose3d(in_channels=64,out_channels=64, kernel_size=4, stride=2, padding=1))
                    self.conv_transpose.append(nn.ReLU())
            
            
              self.conv_transpose4 = nn.ConvTranspose3d(in_channels=64, out_channels=1 , kernel_size=1)

    
     def forward(self,x):
            #import pdb;pdb.set_trace()
            x = self.initial_conv(x)
            x = self.encoded_conv(x)
            x = self.conv_transpose(x)
            x = self.conv_transpose4(x)
            #import pdb;pdb.set_trace()
            return x;

    
    