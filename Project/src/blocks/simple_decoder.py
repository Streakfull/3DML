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
#             self.conva = nn.Conv3d(in_channels=25,out_channels=64,kernel_size=3,padding=1)
#             self.convb = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
#             self.convc = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
#             self.identity = nn.Conv3d(in_channels=25, out_channels=64 ,kernel_size=3,padding=1)
#             self.conva1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
#             self.convb1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
#             self.convc2 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
              self.initial_conv = nn.Conv3d(in_channels=768,out_channels=64, kernel_size=3,padding=1)
              self.encoded_conv = nn.Sequential(ResBlock(64),ResBlock(64))
              
              self.conv_transpose = nn.Sequential()
              for i in range(3):
                    self.conv_transpose.append(nn.ConvTranspose3d(in_channels=64,out_channels=64, kernel_size=4, stride=2, padding=1))
                    self.conv_transpose.append(nn.ReLU())
#               self.conv_transpose1 = nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
#               self.conv_transpose2 = nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
#               self.conv_transpose3 = nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
            
            
              self.conv_transpose4 = nn.ConvTranspose3d(in_channels=64, out_channels=1 , kernel_size=1)

    
     def forward(self,x):
            x = self.initial_conv(x)
            x = self.encoded_conv(x)
            x = self.conv_transpose(x)
            x = self.conv_transpose4(x)
            return x;
      
#          #import pdb;pdb.set_trace()
#          shortcut = self.identity(x)
#          x = self.conva(x)
#          x = self.convb(x)
#          x = self.convc(x)
#          shortcut_2 = x
#          x+=shortcut
#          x = self.conva1(x)
#          x = self.convb1(x)
#          x = self.convc2(x)
#          x+=shortcut_2
#          x = self.conv_transpose1(x)
#          x = self.conv_transpose2(x)
#          x = self.conv_transpose3(x)
#          x = self.conv_transpose4(x)
#          #import pdb;pdb.set_trace()
#          return x
         #import pdb;pdb.set_trace()
    
    