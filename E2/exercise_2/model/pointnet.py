import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        # TODO Add layers: Convolutional k->64, 64->128, 128->1024 with corresponding batch norms and ReLU
        # TODO Add layers: Linear 1024->512, 512->256, 256->k^2 with corresponding batch norms and ReLU

        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k
        conv1 = nn.Conv1d(in_channels=k,out_channels=64,kernel_size=1)
        bn1 = nn.BatchNorm1d(64)
        conv2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        bn2 = nn.BatchNorm1d(128)
        conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        bn3 = nn.BatchNorm1d(1024)
 



        linear1= nn.Linear(1024,512)
        bn_linear_1 = nn.BatchNorm1d(512)
        linear2 = nn.Linear(512,256)
        bn_linear_2 = nn.BatchNorm1d(256)
        linear3 = nn.Linear(256,k**2)
        # bn_linear_3 = nn.BatchNorm1d(k**2)

        self.conv = nn.Sequential(
                conv1,bn1,nn.ReLU(),conv2,bn2,nn.ReLU(), conv3,bn3,
        )

        self.classify = nn.Sequential(linear1,bn_linear_1,nn.ReLU(),linear2,bn_linear_2,nn.ReLU(),linear3)


    def forward(self, x):
        b = x.shape[0]

        # TODO Pass input through layers, applying the same max operation as in PointNetEncoder
        # TODO No batch norm and relu after the 
     
        x = self.conv(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
       
        x = self.classify(x)
        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False):
        super().__init__()

        # TODO Define convolution layers, batch norm layers, and ReLU

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)
        conv1 = nn.Conv1d(in_channels=3,out_channels=64,kernel_size=1)
        bn1 = nn.BatchNorm1d(64)
        self.layer1 = nn.Sequential(conv1,bn1,nn.ReLU())

        conv2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1)
        bn2 = nn.BatchNorm1d(128)

        self.layer2 = nn.Sequential(conv2,bn2,nn.ReLU())
        
        conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        bn3 = nn.BatchNorm1d(1024)
 
        self.layer3 = nn.Sequential(conv3,bn3)



        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]
    
        input_transform = self.input_transform_net(x)

        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)
       
        # TODO: First layer: 3->64
        x  = self.layer1(x)

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        # TODO: Layers 2 and 3: 64->128, 128->1024
        x = self.layer2(x)
   
        x = self.layer3(x)

        # This is the symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = PointNetEncoder(return_point_features=False)
        # TODO Add Linear layers, batch norms, dropout with p=0.3, and ReLU
        # Batch Norms and ReLUs are used after all but the last layer
        # Dropout is used only directly after the second Linear layer
        # The last Linear layer reduces the number of feature channels to num_classes (=k in the architecture visualization)
        linear1 = nn.Linear(1024,512)
        bn1 = nn.BatchNorm1d(512)

        linear2 = nn.Linear(512,256)
        drop_out = nn.Dropout(p=0.3)
        bn2 = nn.BatchNorm1d(256)

        linear3 = nn.Linear(256,num_classes)

        self.classify = nn.Sequential(linear1,bn1,nn.ReLU(),linear2,drop_out, bn2,nn.ReLU(), linear3)
    def forward(self, x):
        x = self.encoder(x)
  
        x = self.classify(x)
        return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(return_point_features=True)
        # TODO: Define convolutions, batch norms, and ReLU

    def forward(self, x):
        x = self.encoder(x)
        # TODO: Pass x through all layers, no batch norm or ReLU after the last conv layer
        x = x.transpose(2, 1).contiguous()
        return x
