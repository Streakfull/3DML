import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        conv1 = nn.Conv3d(in_channels=2, out_channels=self.num_features, kernel_size=4,stride=2,padding=1)
        relu1 = nn.LeakyReLU(negative_slope=0.2)
        conv2 =  nn.Conv3d(in_channels=self.num_features,out_channels=self.num_features*2, kernel_size=4,stride=2,padding=1)
        batch2 = nn.BatchNorm3d(self.num_features*2)
        conv3  =  nn.Conv3d(in_channels=self.num_features*2,out_channels=self.num_features*4, kernel_size=4,stride=2,padding=1)
        batch3 = nn.BatchNorm3d(self.num_features*4)
        conv4  =  nn.Conv3d(in_channels=self.num_features*4,out_channels=self.num_features*8, kernel_size=4,stride=1)
        batch4 = nn.BatchNorm3d(self.num_features*8)

        self.encoder1 = nn.Sequential(conv1,nn.LeakyReLU(negative_slope=0.2))
        self.encoder2 = nn.Sequential(conv2,batch2,nn.LeakyReLU(negative_slope=0.2))
        self.encoder3 = nn.Sequential(conv3,batch3,nn.LeakyReLU(negative_slope=0.2))
        self.encoder4 = nn.Sequential(conv4,batch4,nn.LeakyReLU(negative_slope=0.2))
        # TODO: 2 Bottleneck layers
        linear1 = nn.Linear(in_features=self.num_features*8,out_features=self.num_features*8)
        linear2 = nn.Linear(in_features=self.num_features*8,out_features=self.num_features*8)
        self.bottleneck = nn.Sequential(linear1,nn.ReLU(),linear2,nn.ReLU())

        # TODO: 4 Decoder layers
        conv1 = nn.ConvTranspose3d(in_channels=self.num_features * 8 * 2,out_channels=self.num_features*4,  kernel_size=4)
        batch_norm1 = nn.BatchNorm3d(self.num_features * 4)
        conv2 = nn.ConvTranspose3d(in_channels=self.num_features * 4 * 2,out_channels=self.num_features*2,  kernel_size=4,stride=2,padding=1)
        batch_norm2 = nn.BatchNorm3d(self.num_features * 2)
        conv3 = nn.ConvTranspose3d(in_channels=self.num_features * 2 * 2,out_channels=self.num_features,  kernel_size=4,stride=2,padding=1)
        batch_norm3 = nn.BatchNorm3d(self.num_features)
        conv4 = nn.ConvTranspose3d(in_channels=self.num_features * 2,out_channels=1,kernel_size=4,stride=2,padding=1)

        self.decoder1 = nn.Sequential(conv1,batch_norm1,nn.ReLU())
        self.decoder2 = nn.Sequential(conv2,batch_norm2,nn.ReLU())
        self.decoder3 = nn.Sequential(conv3,batch_norm3,nn.ReLU())
        self.decoder4 = nn.Sequential(conv4)




    def forward(self, x):
        b = x.shape[0]

        x_e1 = self.encoder1(x)
        x_e2 = self.encoder2(x_e1)
        x_e3 = self.encoder3(x_e2)
        x_e4 = self.encoder4(x_e3)

        # print(x_e1.shape)
        # print(x_e2.shape)
        # print(x_e3.shape)

        # # Encode
        # # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        # # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        u1= torch.cat((x,x_e4),dim = 1)
        x_d1 = self.decoder1(u1)
        u2 = torch.cat((x_d1,x_e3),dim=1)
        x_d2 = self.decoder2(u2)
        u3 = torch.cat((x_d2,x_e2),dim=1)
        x_d3 =self.decoder3(u3)
        u4 = torch.cat((x_d3,x_e1),dim=1)
        x = self.decoder4(u4)
        x = torch.squeeze(x, dim=1)
        x = torch.log(torch.abs(x)+1)
        # TODO: Log scaling

        return x
