import torch
from torch import nn
from blocks.residual_block import ResidualBlock

class SimpleDecoder(nn.Module):
     def __init__(self, configs):
              super().__init__()
              d_model = configs["d_model"]
              out_channels = configs["num_pos_embeddings"]
              self.initial_conv = nn.Conv3d(in_channels=d_model,out_channels=out_channels, kernel_size=3,padding=1)
              self.encoded_conv = nn.Sequential(ResidualBlock(out_channels),ResidualBlock(out_channels))
              
              self.conv_transpose = nn.Sequential()
              for i in range(3):
                    self.conv_transpose.append(nn.ConvTranspose3d(in_channels=out_channels,out_channels=out_channels, kernel_size=4, stride=2, padding=1))
                    self.conv_transpose.append(nn.ReLU())
            
            
              self.conv_transpose4 = nn.ConvTranspose3d(in_channels=out_channels, out_channels=1 , kernel_size=1)

    
     def forward(self,x):
            x = self.initial_conv(x)
            x = self.encoded_conv(x)
            x = self.conv_transpose(x)
            x = self.conv_transpose4(x)
            return x

    
    