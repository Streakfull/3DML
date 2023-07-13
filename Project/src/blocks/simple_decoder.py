import torch
from torch import nn


# TODO: Refactor this properly
class SimpleDecoder(nn.Module):
     def __init__(self):
            super().__init__()
            self.conva = nn.Conv3d(in_channels=768,out_channels=64,kernel_size=3,padding=1)
            self.convb = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
            self.convc = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
            self.identity = nn.Conv3d(in_channels=768, out_channels=64 ,kernel_size=3,padding=1)
            self.conva1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
            self.convb1 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
            self.convc2 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
            self.conv_transpose1 = nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
            self.conv_transpose2 = nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
            self.conv_transpose3 = nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1)
            self.conv_transpose4 = nn.ConvTranspose3d(in_channels=64,out_channels=1,kernel_size=1)

    
     def forward(self,x):
         #import pdb;pdb.set_trace()
         shortcut = self.identity(x)
         x = self.conva(x)
         x = self.convb(x)
         x = self.convc(x)
         shortcut_2 = x
         x+=shortcut
         x = self.conva1(x)
         x = self.convb1(x)
         x = self.convc2(x)
         x+=shortcut_2
         x = self.conv_transpose1(x)
         x = self.conv_transpose2(x)
         x = self.conv_transpose3(x)
         x = self.conv_transpose4(x)
         return x
         #import pdb;pdb.set_trace()
    
    